"""HierStore: sqlite hard-tree fact-bank executor (family A).

Ops: ADD / UPDATE / DELETE / NOOP / MOVE / MKDIR.
Soft-delete default; hard DELETE optional. Depth cap on folder path.
Dumb executor — managers/oracles choose ops.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Legacy meta key from v0 ingest (AlwaysADD). Store itself is policy-agnostic.
CONFLICT_POLICY = "append_only"

# Folder segments only (leaf fact not counted). Default for new stores;
# an existing store's META['MAX_DEPTH'] is the source of truth on reopen.
MAX_DEPTH = 5


def max_depth_from_store(db_path: Path | str, default: int = MAX_DEPTH) -> int:
    """Read the depth cap the store was built with. Reopening with the
    constructor default would silently reject deeper gold paths."""
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT value FROM meta WHERE key = 'MAX_DEPTH'"
        ).fetchone()
    finally:
        con.close()
    return int(row[0]) if row else default

# v0 defaults until user-dir snapshot replaces via set_roots().
DEFAULT_ROOTS = ("work", "personal", "inbox", "project", "global")

OPS = frozenset({"ADD", "UPDATE", "DELETE", "NOOP", "MOVE", "MKDIR"})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_path(path: list[str] | None) -> list[str]:
    if not path:
        return []
    out: list[str] = []
    for seg in path:
        s = str(seg).strip().strip("/")
        if not s or s in (".", ".."):
            raise ValueError(f"invalid path segment: {seg!r}")
        out.append(s)
    return out


def path_key(path: list[str]) -> str:
    return "/" + "/".join(path) if path else "/"


def path_prefixes(path: list[str]) -> list[list[str]]:
    """All prefix dirs of path, including path itself."""
    return [path[: i + 1] for i in range(len(path))]


@dataclass
class Op:
    """One memory operation for the executor."""

    op: str  # ADD | UPDATE | DELETE | NOOP | MOVE | MKDIR
    fact_id: str | None = None
    text: str | None = None
    path: list[str] | None = None
    project: str | None = None
    kind: str = "claim"
    t: str | None = None
    episode_id: str | None = None
    parent_id: str | None = None
    target_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    retrieve_ids: list[str] = field(default_factory=list)
    hard: bool = False  # DELETE only
    confidence: float | None = None  # optional header field (leaf B later)

    def normalized(self) -> "Op":
        op = self.op.upper().replace("NO_OPERATION", "NOOP")
        if op not in OPS:
            raise ValueError(f"unknown op: {self.op}")
        path = normalize_path(self.path) if self.path is not None else None
        return Op(
            op=op,
            fact_id=self.fact_id,
            text=self.text,
            path=path,
            project=self.project,
            kind=self.kind,
            t=self.t,
            episode_id=self.episode_id,
            parent_id=self.parent_id,
            target_ids=list(self.target_ids),
            reason=self.reason,
            retrieve_ids=list(self.retrieve_ids),
            hard=self.hard,
            confidence=self.confidence,
        )


class HierStore:
    """Persistent hierarchical fact store — dumb hard-tree executor."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        roots: list[str] | tuple[str, ...] | None = None,
        max_depth: int = MAX_DEPTH,
        strict_dirs: bool = False,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.max_depth = int(max_depth)
        self.strict_dirs = bool(strict_dirs)
        self._init_schema()
        stored_depth = self.get_meta("MAX_DEPTH")
        if stored_depth is not None:
            self.max_depth = int(stored_depth)
        if roots is not None:
            self.set_roots(list(roots))
        elif self.get_meta("FIXED_ROOTS") is None:
            self.set_roots(list(DEFAULT_ROOTS))
        self._seed_root_dirs()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dirs (
              path_key TEXT PRIMARY KEY,
              path_json TEXT NOT NULL,
              depth INTEGER NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_dirs_depth ON dirs(depth);
            CREATE TABLE IF NOT EXISTS facts (
              id TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              path_json TEXT NOT NULL,
              project TEXT,
              kind TEXT NOT NULL,
              t TEXT,
              valid INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              revision INTEGER NOT NULL DEFAULT 1,
              parent_id TEXT,
              episode_id TEXT,
              confidence REAL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project);
            CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid);
            CREATE TABLE IF NOT EXISTS embeddings (
              id TEXT NOT NULL,
              model TEXT NOT NULL,
              dim INTEGER NOT NULL,
              vector BLOB NOT NULL,
              PRIMARY KEY (id, model),
              FOREIGN KEY (id) REFERENCES facts(id)
            );
            CREATE TABLE IF NOT EXISTS ops_log (
              op_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              op TEXT NOT NULL,
              fact_id TEXT,
              target_ids_json TEXT NOT NULL,
              before_text TEXT,
              after_text TEXT,
              project TEXT,
              path_json TEXT,
              manager TEXT,
              reason TEXT,
              retrieve_ids_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ops_log_ts ON ops_log(ts);
            """
        )
        self._migrate_facts_columns()
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("CONFLICT_POLICY", CONFLICT_POLICY),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("ARCHITECTURE", "family_A_hard_tree"),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("MAX_DEPTH", str(self.max_depth)),
        )
        self.conn.commit()

    def _migrate_facts_columns(self) -> None:
        cols = {
            r["name"]
            for r in self.conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        alters = []
        if "revision" not in cols:
            alters.append(
                "ALTER TABLE facts ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
            )
        if "parent_id" not in cols:
            alters.append("ALTER TABLE facts ADD COLUMN parent_id TEXT")
        if "episode_id" not in cols:
            alters.append("ALTER TABLE facts ADD COLUMN episode_id TEXT")
        if "confidence" not in cols:
            alters.append("ALTER TABLE facts ADD COLUMN confidence REAL")
        for sql in alters:
            self.conn.execute(sql)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HierStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_roots(self, roots: list[str]) -> None:
        roots_n = normalize_path(roots)
        if not roots_n:
            raise ValueError("roots must be non-empty")
        self.set_meta("FIXED_ROOTS", json.dumps(roots_n))
        self._seed_root_dirs()

    def roots(self) -> list[str]:
        raw = self.get_meta("FIXED_ROOTS")
        if not raw:
            return list(DEFAULT_ROOTS)
        return list(json.loads(raw))

    def _seed_root_dirs(self) -> None:
        for r in self.roots():
            self._ensure_dir([r], commit=False)
        self.conn.commit()

    def validate_path(self, path: list[str], *, as_dir: bool = True) -> list[str]:
        path = normalize_path(path)
        if as_dir and not path:
            raise ValueError("empty path not allowed")
        if len(path) > self.max_depth:
            raise ValueError(
                f"path depth {len(path)} exceeds max_depth={self.max_depth}: {path}"
            )
        roots = self.roots()
        if path[0] not in roots:
            raise ValueError(
                f"path root {path[0]!r} not in fixed roots {roots}"
            )
        return path

    def dir_exists(self, path: list[str]) -> bool:
        path = normalize_path(path)
        row = self.conn.execute(
            "SELECT 1 FROM dirs WHERE path_key = ?", (path_key(path),)
        ).fetchone()
        return row is not None

    def _ensure_dir(self, path: list[str], *, commit: bool) -> list[str]:
        path = self.validate_path(path, as_dir=True)
        now = _now()
        for pref in path_prefixes(path):
            pk = path_key(pref)
            self.conn.execute(
                """
                INSERT OR IGNORE INTO dirs(path_key, path_json, depth, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (pk, json.dumps(pref), len(pref), now),
            )
        if commit:
            self.conn.commit()
        return path

    def mkdir(self, path: list[str], *, commit: bool = True) -> list[str]:
        """Create directory path (mkdir -p) under fixed roots / depth cap."""
        return self._ensure_dir(path, commit=commit)

    def list_dirs(self) -> list[list[str]]:
        rows = self.conn.execute(
            "SELECT path_json FROM dirs ORDER BY depth, path_key"
        ).fetchall()
        return [json.loads(r["path_json"]) for r in rows]

    def snapshot(self) -> dict[str, Any]:
        """Gold-comparable tree snapshot (dirs + facts)."""
        facts = []
        for f in self.read_all(valid_only=False):
            facts.append(
                {
                    "id": f["id"],
                    "text": f["text"],
                    "path": f["path"],
                    "valid": f["valid"],
                    "project": f["project"],
                    "confidence": f.get("confidence"),
                }
            )
        facts.sort(key=lambda x: x["id"])
        dirs = sorted(self.list_dirs(), key=lambda p: path_key(p))
        return {
            "dirs": dirs,
            "facts": facts,
            "valid_count": sum(1 for f in facts if f["valid"]),
            "roots": self.roots(),
            "max_depth": self.max_depth,
        }

    def create(self, fact: dict[str, Any]) -> str:
        """Insert a new fact. Auto-ensures path dirs (legacy AlwaysADD-friendly)."""
        return self._insert_fact(fact, commit=True, ensure_dirs=True)

    def _default_path(self, project: str | None) -> list[str]:
        if project is None:
            return ["global"]
        return ["project", project]

    def _insert_fact(
        self, fact: dict[str, Any], commit: bool, *, ensure_dirs: bool
    ) -> str:
        fid = fact["id"]
        path = normalize_path(
            fact.get("path")
            or self._default_path(fact.get("project"))
        )
        path = self.validate_path(path, as_dir=True)
        if ensure_dirs:
            self._ensure_dir(path, commit=False)
        elif not self.dir_exists(path):
            raise ValueError(f"ADD path does not exist (MKDIR first): {path}")
        now = _now()
        try:
            self.conn.execute(
                """
                INSERT INTO facts(
                  id, text, path_json, project, kind, t, valid,
                  created_at, updated_at, revision, parent_id, episode_id,
                  confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, 1, ?, ?, ?)
                """,
                (
                    fid,
                    fact["text"],
                    json.dumps(path),
                    fact.get("project"),
                    fact.get("kind", "claim"),
                    fact.get("t"),
                    now,
                    now,
                    fact.get("parent_id"),
                    fact.get("episode_id"),
                    fact.get("confidence"),
                ),
            )
            if commit:
                self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"duplicate fact id: {fid}") from e
        return fid

    def get(self, fact_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return self._row_to_fact(row) if row else None

    def read_all(self, valid_only: bool = True) -> list[dict[str, Any]]:
        if valid_only:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE valid = 1 ORDER BY created_at, id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM facts ORDER BY created_at, id"
            ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def read_by_project(
        self, project: str | None, valid_only: bool = True
    ) -> list[dict[str, Any]]:
        if project is None:
            q = "SELECT * FROM facts WHERE project IS NULL"
        else:
            q = "SELECT * FROM facts WHERE project = ?"
        if valid_only:
            q += " AND valid = 1"
        q += " ORDER BY created_at, id"
        if project is None:
            rows = self.conn.execute(q).fetchall()
        else:
            rows = self.conn.execute(q, (project,)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def read_subtree(
        self, active_path: list[str], valid_only: bool = True
    ) -> list[dict[str, Any]]:
        """Facts whose path equals active_path or is under it (descendants)."""
        active = normalize_path(active_path)
        if not active:
            return self.read_all(valid_only=valid_only)
        out = []
        for f in self.read_all(valid_only=valid_only):
            p = f["path"]
            if p[: len(active)] == active:
                out.append(f)
        return out

    def update(self, fact_id: str, **fields: Any) -> dict[str, Any]:
        """In-place UPDATE. Clears embeddings for this id (caller re-embeds)."""
        before = self.get(fact_id)
        if before is None:
            raise KeyError(f"fact not found: {fact_id}")
        if not before["valid"]:
            raise ValueError(f"cannot update soft-deleted fact: {fact_id}")

        text = fields.get("text", before["text"])
        path = fields.get("path", before["path"])
        if "path" in fields:
            path = self.validate_path(normalize_path(path), as_dir=True)
            if not self.dir_exists(path):
                self._ensure_dir(path, commit=False)
        project = fields["project"] if "project" in fields else before["project"]
        kind = fields.get("kind", before["kind"])
        t = fields["t"] if "t" in fields else before["t"]
        parent_id = (
            fields["parent_id"] if "parent_id" in fields else before.get("parent_id")
        )
        episode_id = (
            fields["episode_id"]
            if "episode_id" in fields
            else before.get("episode_id")
        )
        confidence = (
            fields["confidence"]
            if "confidence" in fields
            else before.get("confidence")
        )
        now = _now()
        self.conn.execute(
            """
            UPDATE facts SET
              text = ?, path_json = ?, project = ?, kind = ?, t = ?,
              updated_at = ?, revision = revision + 1,
              parent_id = ?, episode_id = ?, confidence = ?
            WHERE id = ?
            """,
            (
                text,
                json.dumps(path),
                project,
                kind,
                t,
                now,
                parent_id,
                episode_id,
                confidence,
                fact_id,
            ),
        )
        if text != before["text"]:
            self.conn.execute("DELETE FROM embeddings WHERE id = ?", (fact_id,))
        self.conn.commit()
        after = self.get(fact_id)
        assert after is not None
        return after

    def move(self, fact_id: str, new_path: list[str], *, commit: bool = True) -> dict[str, Any]:
        """Re-parent fact; text unchanged. Target dir must exist unless not strict."""
        before = self.get(fact_id)
        if before is None:
            raise KeyError(f"fact not found: {fact_id}")
        if not before["valid"]:
            raise ValueError(f"cannot MOVE soft-deleted fact: {fact_id}")
        new_path = self.validate_path(normalize_path(new_path), as_dir=True)
        if not self.dir_exists(new_path):
            if self.strict_dirs:
                raise ValueError(f"MOVE target dir missing (MKDIR first): {new_path}")
            self._ensure_dir(new_path, commit=False)
        now = _now()
        project = new_path[1] if new_path[0] == "project" and len(new_path) > 1 else before["project"]
        if new_path[0] in ("work", "personal", "inbox") and len(new_path) > 1:
            project = new_path[1]
        self.conn.execute(
            """
            UPDATE facts SET path_json = ?, project = ?, updated_at = ?,
              revision = revision + 1
            WHERE id = ?
            """,
            (json.dumps(new_path), project, now, fact_id),
        )
        if commit:
            self.conn.commit()
        after = self.get(fact_id)
        assert after is not None
        return after

    def delete(self, fact_id: str, hard: bool = False) -> None:
        row = self.get(fact_id)
        if row is None:
            raise KeyError(f"fact not found: {fact_id}")
        if hard:
            self.conn.execute("DELETE FROM embeddings WHERE id = ?", (fact_id,))
            self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        else:
            self.conn.execute(
                "UPDATE facts SET valid = 0, updated_at = ? WHERE id = ?",
                (_now(), fact_id),
            )
        self.conn.commit()

    def apply_ops(
        self,
        ops: Iterable[Op | dict[str, Any]],
        *,
        manager: str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply a batch of ops in one transaction. Returns ops_log rows."""
        parsed = [
            (Op(**o) if isinstance(o, dict) else o).normalized() for o in ops
        ]
        log_rows: list[dict[str, Any]] = []
        try:
            for op in parsed:
                log_rows.append(self._apply_one(op, manager=manager, commit=False))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return log_rows

    def _apply_one(
        self, op: Op, *, manager: str | None, commit: bool
    ) -> dict[str, Any]:
        before_text: str | None = None
        after_text: str | None = None
        fact_id = op.fact_id
        project = op.project
        path = op.path
        target_ids = list(op.target_ids)

        if op.op == "MKDIR":
            if not op.path:
                raise ValueError("MKDIR requires path")
            path = self._ensure_dir(op.path, commit=False)
            after_text = path_key(path)

        elif op.op == "MOVE":
            if not fact_id:
                raise ValueError("MOVE requires fact_id")
            if not op.path:
                raise ValueError("MOVE requires path (destination)")
            before = self.get(fact_id)
            if before is None:
                raise KeyError(f"MOVE target missing: {fact_id}")
            before_text = before["text"]
            after = self.move(fact_id, op.path, commit=False)
            after_text = before_text
            path = after["path"]
            project = after["project"]
            if fact_id not in target_ids:
                target_ids = [fact_id] + target_ids

        elif op.op == "ADD":
            if not op.text:
                raise ValueError("ADD requires text")
            fact_id = fact_id or f"f_{uuid.uuid4().hex[:12]}"
            path = path or self._default_path(project)
            path = self.validate_path(normalize_path(path), as_dir=True)
            ensure = not self.strict_dirs
            if self.strict_dirs and not self.dir_exists(path):
                raise ValueError(f"ADD path does not exist (MKDIR first): {path}")
            self._insert_fact(
                {
                    "id": fact_id,
                    "text": op.text,
                    "path": path,
                    "project": project,
                    "kind": op.kind,
                    "t": op.t or _now(),
                    "parent_id": op.parent_id,
                    "episode_id": op.episode_id,
                    "confidence": op.confidence,
                },
                commit=False,
                ensure_dirs=ensure,
            )
            after_text = op.text
            if fact_id not in target_ids:
                target_ids = [fact_id] + target_ids

        elif op.op == "UPDATE":
            if not fact_id:
                raise ValueError("UPDATE requires fact_id")
            before = self.get(fact_id)
            if before is None:
                raise KeyError(f"UPDATE target missing: {fact_id}")
            before_text = before["text"]
            fields: dict[str, Any] = {}
            if op.text is not None:
                fields["text"] = op.text
            if op.path is not None:
                fields["path"] = self.validate_path(op.path, as_dir=True)
                if not self.dir_exists(fields["path"]):
                    if self.strict_dirs:
                        raise ValueError(
                            f"UPDATE path missing (MKDIR first): {fields['path']}"
                        )
                    self._ensure_dir(fields["path"], commit=False)
            if op.project is not None or (op.path and op.path == ["global"]):
                fields["project"] = op.project
            if op.confidence is not None:
                fields["confidence"] = op.confidence
            text = fields.get("text", before["text"])
            path_u = fields.get("path", before["path"])
            project_u = fields["project"] if "project" in fields else before["project"]
            conf_u = (
                fields["confidence"]
                if "confidence" in fields
                else before.get("confidence")
            )
            now = _now()
            self.conn.execute(
                """
                UPDATE facts SET
                  text = ?, path_json = ?, project = ?,
                  updated_at = ?, revision = revision + 1,
                  parent_id = COALESCE(?, parent_id),
                  episode_id = COALESCE(?, episode_id),
                  confidence = ?
                WHERE id = ?
                """,
                (
                    text,
                    json.dumps(path_u),
                    project_u,
                    now,
                    op.parent_id,
                    op.episode_id,
                    conf_u,
                    fact_id,
                ),
            )
            if text != before_text:
                self.conn.execute("DELETE FROM embeddings WHERE id = ?", (fact_id,))
            after_text = text
            project = project_u
            path = path_u
            if fact_id not in target_ids:
                target_ids = [fact_id] + target_ids

        elif op.op == "DELETE":
            if not fact_id:
                raise ValueError("DELETE requires fact_id")
            before = self.get(fact_id)
            if before is None:
                raise KeyError(f"DELETE target missing: {fact_id}")
            before_text = before["text"]
            project = before["project"]
            path = before["path"]
            if op.hard:
                self.conn.execute("DELETE FROM embeddings WHERE id = ?", (fact_id,))
                self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            else:
                self.conn.execute(
                    "UPDATE facts SET valid = 0, updated_at = ? WHERE id = ?",
                    (_now(), fact_id),
                )
            if fact_id not in target_ids:
                target_ids = [fact_id] + target_ids

        elif op.op == "NOOP":
            pass

        log_row = self._log_op(
            op=op.op,
            fact_id=fact_id,
            target_ids=target_ids,
            before_text=before_text,
            after_text=after_text,
            project=project,
            path=path,
            manager=manager,
            reason=op.reason,
            retrieve_ids=op.retrieve_ids,
            commit=commit,
        )
        return log_row

    def _log_op(
        self,
        *,
        op: str,
        fact_id: str | None,
        target_ids: list[str],
        before_text: str | None,
        after_text: str | None,
        project: str | None,
        path: list[str] | None,
        manager: str | None,
        reason: str | None,
        retrieve_ids: list[str],
        commit: bool,
    ) -> dict[str, Any]:
        op_id = f"op_{uuid.uuid4().hex[:16]}"
        ts = _now()
        row = {
            "op_id": op_id,
            "ts": ts,
            "op": op,
            "fact_id": fact_id,
            "target_ids": target_ids,
            "before_text": before_text,
            "after_text": after_text,
            "project": project,
            "path": path,
            "manager": manager,
            "reason": reason,
            "retrieve_ids": retrieve_ids,
        }
        self.conn.execute(
            """
            INSERT INTO ops_log(
              op_id, ts, op, fact_id, target_ids_json, before_text, after_text,
              project, path_json, manager, reason, retrieve_ids_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                op_id,
                ts,
                op,
                fact_id,
                json.dumps(target_ids),
                before_text,
                after_text,
                project,
                json.dumps(path) if path is not None else None,
                manager,
                reason,
                json.dumps(retrieve_ids),
            ),
        )
        if commit:
            self.conn.commit()
        return row

    def read_ops_log(self, limit: int | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM ops_log ORDER BY ts, op_id"
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        rows = self.conn.execute(q).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "op_id": r["op_id"],
                    "ts": r["ts"],
                    "op": r["op"],
                    "fact_id": r["fact_id"],
                    "target_ids": json.loads(r["target_ids_json"] or "[]"),
                    "before_text": r["before_text"],
                    "after_text": r["after_text"],
                    "project": r["project"],
                    "path": json.loads(r["path_json"]) if r["path_json"] else None,
                    "manager": r["manager"],
                    "reason": r["reason"],
                    "retrieve_ids": json.loads(r["retrieve_ids_json"] or "[]"),
                }
            )
        return out

    def put_embedding(self, fact_id: str, model: str, vector: list[float] | Any) -> None:
        import numpy as np

        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        blob = arr.tobytes()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings(id, model, dim, vector)
            VALUES (?, ?, ?, ?)
            """,
            (fact_id, model, int(arr.shape[0]), blob),
        )
        self.conn.commit()

    def get_embedding(self, fact_id: str, model: str) -> list[float] | None:
        import numpy as np

        row = self.conn.execute(
            "SELECT dim, vector FROM embeddings WHERE id = ? AND model = ?",
            (fact_id, model),
        ).fetchone()
        if not row:
            return None
        arr = np.frombuffer(row["vector"], dtype=np.float32)
        return arr.tolist()

    def delete_embedding(self, fact_id: str, model: str | None = None) -> None:
        if model is None:
            self.conn.execute("DELETE FROM embeddings WHERE id = ?", (fact_id,))
        else:
            self.conn.execute(
                "DELETE FROM embeddings WHERE id = ? AND model = ?",
                (fact_id, model),
            )
        self.conn.commit()

    def load_embeddings_matrix(
        self, facts: list[dict[str, Any]], model: str
    ) -> Any:
        """Return list of float32 vectors aligned with facts (None if missing)."""
        import numpy as np

        rows = self.conn.execute(
            "SELECT id, vector FROM embeddings WHERE model = ?",
            (model,),
        ).fetchall()
        by_id = {
            r["id"]: np.frombuffer(r["vector"], dtype=np.float32).copy() for r in rows
        }
        return [by_id.get(f["id"]) for f in facts]

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        return {
            "id": row["id"],
            "text": row["text"],
            "path": json.loads(row["path_json"]),
            "project": row["project"],
            "kind": row["kind"],
            "t": row["t"],
            "valid": bool(row["valid"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "revision": int(row["revision"]) if "revision" in keys else 1,
            "parent_id": row["parent_id"] if "parent_id" in keys else None,
            "episode_id": row["episode_id"] if "episode_id" in keys else None,
            "confidence": row["confidence"] if "confidence" in keys else None,
            "tags": [],
        }

    def conflict_policy(self) -> str:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'CONFLICT_POLICY'"
        ).fetchone()
        return row["value"] if row else CONFLICT_POLICY
