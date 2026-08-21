#!/usr/bin/env python3
"""Figure 1: accuracy by support stratum, corpus B, item-stratified split."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"

BUCKETS = ["1–2", "3–9", "10+"]
X = [0, 1, 2]
N = [258, 653, 558]

DATA = {
    "flat (gpt-4o)": {"y": [0.547, 0.662, 0.332], "color": "#6b7280", "marker": "s", "ls": "--"},
    "BM25":          {"y": [0.403, 0.636, 0.771], "color": "#0f766e", "marker": "D", "ls": "-."},
    "kNN":           {"y": [0.457, 0.718, 0.833], "color": "#2563eb", "marker": "o", "ls": "-"},
    "cascade":       {"y": [0.628, 0.729, 0.414], "color": "#d97706", "marker": "v", "ls": "-"},
    "LoRA":          {"y": [0.667, 0.776, 0.772], "color": "#b91c1c", "marker": "^", "ls": "-"},
}


def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# outlier-trimmed series (see scripts/occupancy_sensitivity.py, 3 vaults removed)
TRIM = {
    "cascade": ([0.629, 0.732, 0.621], "#d97706"),
    "kNN":     ([0.454, 0.710, 0.776], "#2563eb"),
}

fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=200)

for label, (y, color) in TRIM.items():
    ax.plot(X, y, color=color, linestyle=":", linewidth=1.4, alpha=0.55,
            marker="", zorder=2)

for label, d in DATA.items():
    y = d["y"]
    lo_err, hi_err = [], []
    for yi, n in zip(y, N):
        lo, hi = wilson_ci(yi, n)
        lo_err.append(yi - lo)
        hi_err.append(hi - yi)
    ax.errorbar(
        X,
        y,
        yerr=[lo_err, hi_err],
        color=d["color"],
        marker=d["marker"],
        linestyle=d["ls"],
        linewidth=2.0,
        markersize=7,
        label=label,
        capsize=3,
        elinewidth=1.1,
        zorder=3,
    )

ax.set_xticks(X)
ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BUCKETS, N)], fontsize=9)
ax.set_xlabel("training notes already in the gold folder", fontsize=10)
ax.set_ylabel("exact-match accuracy", fontsize=10)
ax.set_ylim(0.25, 0.95)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_yticklabels([".3", ".4", ".5", ".6", ".7", ".8", ".9"], fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
ax.legend(loc="lower right", frameon=False, fontsize=8.5, handlelength=2.4, ncol=1)
ax.plot([], [], color="#6b7280", linestyle=":", linewidth=1.4, alpha=0.7,
        label="same, 3 outlier vaults removed")
ax.set_title("Corpus B, item-stratified split (27 vaults, 95% CI)", fontsize=10.5, pad=8)

fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig1_coverage_curve.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_coverage_curve.png", bbox_inches="tight")
print("wrote", OUT / "fig1_coverage_curve.pdf")
