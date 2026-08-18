"""Unit tests for HierStore CRUD executor + managers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.manager import (
    AlwaysADD,
    ManagerInput,
    RuleV0,
    get_manager,
    parse_llm_ops,
)
from harness.store import CONFLICT_POLICY, HierStore, Op


class TestHierStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite"
        self.store = HierStore(self.db)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_create_and_get(self) -> None:
        fid = self.store.create(
            {
                "id": "f1",
                "text": "hello",
                "path": ["project", "slm-lab"],
                "project": "slm-lab",
                "kind": "claim",
                "t": "2026-01-01T00:00:00Z",
            }
        )
        self.assertEqual(fid, "f1")
        got = self.store.get("f1")
        assert got is not None
        self.assertEqual(got["text"], "hello")
        self.assertEqual(got["project"], "slm-lab")
        self.assertEqual(got["path"], ["project", "slm-lab"])
        self.assertEqual(got["revision"], 1)

    def test_duplicate_id_raises(self) -> None:
        fact = {
            "id": "f1",
            "text": "a",
            "project": "p",
            "kind": "claim",
            "t": None,
        }
        self.store.create(fact)
        with self.assertRaises(ValueError):
            self.store.create({**fact, "text": "b"})

    def test_append_only_keeps_contrasting(self) -> None:
        self.store.create(
            {"id": "a", "text": "F1=0.50 on X", "project": "p", "kind": "claim", "t": None}
        )
        self.store.create(
            {"id": "b", "text": "F1=0.50 on Y", "project": "p", "kind": "claim", "t": None}
        )
        self.assertEqual(len(self.store.read_by_project("p")), 2)
        self.assertEqual(self.store.conflict_policy(), CONFLICT_POLICY)

    def test_read_by_project_and_global(self) -> None:
        self.store.create(
            {"id": "g1", "text": "prefers TS", "project": None, "kind": "claim", "t": None}
        )
        self.store.create(
            {"id": "p1", "text": "local", "project": "alpha", "kind": "claim", "t": None}
        )
        self.assertEqual(len(self.store.read_by_project(None)), 1)
        self.assertEqual(len(self.store.read_by_project("alpha")), 1)
        self.assertEqual(len(self.store.read_all()), 2)

    def test_update_revision_and_clear_embed(self) -> None:
        self.store.create(
            {"id": "f1", "text": "x", "project": "p", "kind": "claim", "t": None}
        )
        self.store.put_embedding("f1", "m", [0.1, 0.2])
        after = self.store.update("f1", text="y")
        self.assertEqual(after["text"], "y")
        self.assertEqual(after["revision"], 2)
        self.assertIsNone(self.store.get_embedding("f1", "m"))

    def test_soft_delete_hides_from_read(self) -> None:
        self.store.create(
            {"id": "f1", "text": "x", "project": "p", "kind": "claim", "t": None}
        )
        self.store.delete("f1", hard=False)
        self.assertEqual(len(self.store.read_all(valid_only=True)), 0)
        self.assertEqual(len(self.store.read_all(valid_only=False)), 1)
        got = self.store.get("f1")
        assert got is not None
        self.assertFalse(got["valid"])

    def test_hard_delete(self) -> None:
        self.store.create(
            {"id": "f1", "text": "x", "project": "p", "kind": "claim", "t": None}
        )
        self.store.put_embedding("f1", "m", [1.0])
        self.store.delete("f1", hard=True)
        self.assertIsNone(self.store.get("f1"))

    def test_apply_ops_batch_and_log(self) -> None:
        logs = self.store.apply_ops(
            [
                Op(op="ADD", fact_id="a", text="one", project="p"),
                Op(op="ADD", fact_id="b", text="two", project="p"),
                Op(op="UPDATE", fact_id="a", text="one-v2"),
                Op(op="NOOP", fact_id="b", reason="skip"),
                Op(op="DELETE", fact_id="b"),
            ],
            manager="test",
        )
        self.assertEqual(len(logs), 5)
        self.assertEqual([l["op"] for l in logs], ["ADD", "ADD", "UPDATE", "NOOP", "DELETE"])
        self.assertEqual(self.store.get("a")["text"], "one-v2")
        self.assertEqual(len(self.store.read_all(valid_only=True)), 1)
        self.assertEqual(len(self.store.read_ops_log()), 5)

    def test_apply_ops_rollback_on_error(self) -> None:
        self.store.create(
            {"id": "a", "text": "x", "project": "p", "kind": "claim", "t": None}
        )
        with self.assertRaises(KeyError):
            self.store.apply_ops(
                [
                    Op(op="UPDATE", fact_id="a", text="y"),
                    Op(op="UPDATE", fact_id="missing", text="z"),
                ]
            )
        # rolled back — still original text
        self.assertEqual(self.store.get("a")["text"], "x")

    def test_embedding_roundtrip(self) -> None:
        self.store.create(
            {"id": "f1", "text": "x", "project": "p", "kind": "claim", "t": None}
        )
        vec = [0.1, 0.2, 0.3]
        self.store.put_embedding("f1", "text-embedding-3-small", vec)
        got = self.store.get_embedding("f1", "text-embedding-3-small")
        assert got is not None
        self.assertEqual(len(got), 3)
        self.assertAlmostEqual(got[0], 0.1, places=5)

    def test_mkdir_move_subtree(self) -> None:
        self.store.set_roots(["work", "personal", "inbox"])
        self.store.apply_ops(
            [
                Op(op="MKDIR", path=["work", "lab"]),
                Op(op="ADD", fact_id="f1", text="note", path=["work", "lab"]),
                Op(op="MKDIR", path=["personal", "pets"]),
                Op(op="MOVE", fact_id="f1", path=["personal", "pets"]),
            ]
        )
        self.assertEqual(self.store.get("f1")["path"], ["personal", "pets"])
        self.assertEqual(
            [f["id"] for f in self.store.read_subtree(["personal"])],
            ["f1"],
        )
        self.assertEqual(self.store.read_subtree(["work", "lab"]), [])

    def test_depth_cap(self) -> None:
        self.store.set_roots(["work", "personal", "inbox"])
        with self.assertRaises(ValueError):
            self.store.mkdir(["work", "a", "b", "c", "d", "e"])

    def test_strict_add_requires_mkdir(self) -> None:
        self.store.close()
        self.store = HierStore(
            self.db, roots=["work", "personal", "inbox"], strict_dirs=True
        )
        with self.assertRaises(ValueError):
            self.store.apply_ops(
                [Op(op="ADD", fact_id="f1", text="x", path=["work", "missing"])]
            )


class TestManagers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite"
        self.store = HierStore(self.db)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_always_add(self) -> None:
        mgr = AlwaysADD()
        logs = mgr.apply(
            ManagerInput(text="hello", project="p", fact_id="f1"), self.store
        )
        self.assertEqual(logs[0]["op"], "ADD")
        self.assertEqual(self.store.get("f1")["text"], "hello")

    def test_rule_v0_noop_dupe(self) -> None:
        self.store.create(
            {
                "id": "s1",
                "text": "Andrew adopted a dog named Buddy.",
                "project": "p",
                "kind": "claim",
                "t": None,
            }
        )
        emb = {"s1": np.ones(8, dtype=np.float32)}
        emb["s1"] = emb["s1"] / np.linalg.norm(emb["s1"])

        def embed_fn(texts):
            # identical vector for same normalized content
            out = []
            for t in texts:
                if "Buddy" in t and "Scout" not in t:
                    out.append(emb["s1"])
                else:
                    v = np.random.default_rng(abs(hash(t)) % (2**32)).normal(size=8)
                    v = v / np.linalg.norm(v)
                    out.append(v.astype(np.float32))
            return np.stack(out)

        mgr = RuleV0(embeddings=dict(emb), embed_fn=embed_fn, noop_thresh=0.95)
        logs = mgr.apply(
            ManagerInput(
                text="Andrew adopted a dog named Buddy.",
                project="p",
            ),
            self.store,
        )
        self.assertEqual(logs[0]["op"], "NOOP")
        self.assertEqual(len(self.store.read_all()), 1)

    def test_rule_v0_supersede_delete_add(self) -> None:
        self.store.create(
            {
                "id": "s1",
                "text": "Caroline lives in Seattle.",
                "project": "p",
                "kind": "claim",
                "t": None,
            }
        )
        v = np.ones(8, dtype=np.float32)
        v = v / np.linalg.norm(v)

        def embed_fn(texts):
            return np.stack([v for _ in texts])

        mgr = RuleV0(
            embeddings={"s1": v},
            embed_fn=embed_fn,
            update_thresh=0.8,
            noop_thresh=0.999,
        )
        logs = mgr.apply(
            ManagerInput(
                text="Caroline no longer lives in Seattle; she moved to Portland.",
                project="p",
                fact_id="s_new",
            ),
            self.store,
        )
        self.assertEqual([l["op"] for l in logs], ["DELETE", "ADD"])
        self.assertFalse(self.store.get("s1")["valid"])
        self.assertIn("Portland", self.store.get("s_new")["text"])

    def test_rule_v0_update_complement(self) -> None:
        self.store.create(
            {
                "id": "s1",
                "text": "Andrew adopted a dog named Buddy.",
                "project": "p",
                "kind": "claim",
                "t": None,
            }
        )
        v = np.ones(8, dtype=np.float32)
        v = v / np.linalg.norm(v)

        def embed_fn(texts):
            return np.stack([v for _ in texts])

        mgr = RuleV0(
            embeddings={"s1": v},
            embed_fn=embed_fn,
            update_thresh=0.8,
            noop_thresh=1.01,  # only exact-text NOOP; sim=1 still UPDATE
        )
        logs = mgr.apply(
            ManagerInput(
                text="Andrew also adopted a dog named Scout.",
                project="p",
            ),
            self.store,
        )
        self.assertEqual(logs[0]["op"], "UPDATE")
        self.assertIn("Scout", self.store.get("s1")["text"])

    def test_get_manager(self) -> None:
        self.assertIsInstance(get_manager("AlwaysADD"), AlwaysADD)
        self.assertIsInstance(get_manager("RuleV0"), RuleV0)
        from harness.manager import LLMv0

        self.assertIsInstance(get_manager("LLMv0"), LLMv0)

    def test_parse_llm_ops_update(self) -> None:
        related = [{"id": "seed_buddy", "text": "Andrew adopted Buddy."}]
        inp = ManagerInput(text="also Scout", project="p")
        raw = json.dumps(
            [
                {
                    "event": "UPDATE",
                    "id": "seed_buddy",
                    "text": "Andrew adopted Buddy and Scout.",
                }
            ]
        )
        ops = parse_llm_ops(raw, related=related, inp=inp)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].op, "UPDATE")
        self.assertEqual(ops[0].fact_id, "seed_buddy")
        self.assertIn("Scout", ops[0].text or "")

    def test_parse_llm_ops_bad_json_raises_or_fallback(self) -> None:
        from harness.manager import ParseError

        inp = ManagerInput(text="new fact", project="p", fact_id="f_new")
        with self.assertRaises(ParseError):
            parse_llm_ops("not json", related=[], inp=inp, allow_fallback=False)
        ops = parse_llm_ops("not json", related=[], inp=inp, allow_fallback=True)
        self.assertEqual(ops[0].op, "ADD")
        self.assertEqual(ops[0].fact_id, "f_new")


if __name__ == "__main__":
    unittest.main()
