#!/usr/bin/env python3
"""Figure 1: observed accuracy by support stratum, corpus A.

The x=0 point comes from a separately constructed folder-disjoint split
with different method configurations (see paper Methods) -- it is NOT the
left edge of the same fitted curve as x=1..3, so it is plotted disconnected
from the item-split line, not joined to it. Wilson-score 95% CIs shown per
point since the 1-2 bucket is n=18.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKETS = ["0", "1–2", "3–9", "10+"]
X = [0, 1, 2, 3]
N = [353, 18, 105, 171]

DATA = {
    "gpt-4o (flat)":       {"y": [0.354, 0.278, 0.200, 0.339], "color": "#8a8f98", "marker": "s", "ls": "--"},
    "kNN (retrieval)":     {"y": [0.000, 0.444, 0.381, 0.620], "color": "#2f6fed", "marker": "o", "ls": "-"},
    "LoRA (fine-tuned)":   {"y": [0.363, 0.222, 0.371, 0.673], "color": "#c0392b", "marker": "^", "ls": "-"},
}


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=200)

for label, d in DATA.items():
    y = d["y"]
    lo_err, hi_err = [], []
    for yi, n in zip(y, N):
        lo, hi = wilson_ci(yi, n)
        lo_err.append(yi - lo)
        hi_err.append(hi - yi)
    # x=0 plotted disconnected: separate scatter, no line to x=1
    ax.errorbar(X[0], y[0], yerr=[[lo_err[0]], [hi_err[0]]],
                fmt=d["marker"], color=d["color"], markersize=7,
                capsize=3, elinewidth=1.2, zorder=3)
    ax.errorbar(X[1:], y[1:], yerr=[lo_err[1:], hi_err[1:]],
                color=d["color"], marker=d["marker"], linestyle=d["ls"],
                linewidth=2.0, markersize=7, label=label,
                capsize=3, elinewidth=1.2, zorder=3)

# visual break marking "different split, not the same curve"
ax.axvline(0.5, color="#c8ccd3", linestyle=":", linewidth=1.2, zorder=1)
ax.text(0.5, -0.145, "folder-disjoint split   |   item-stratified split",
        ha="center", va="top", fontsize=8.5, color="#6b7280",
        transform=ax.get_xaxis_transform())

ax.set_xticks(X)
ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BUCKETS, N)], fontsize=9)
ax.set_xlabel("training notes already in the gold folder", fontsize=10, labelpad=22)
ax.set_ylabel("placement exact-match accuracy", fontsize=10)
ax.set_ylim(-0.05, 0.90)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0", ".2", ".4", ".6", ".8"], fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)

ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.2)
ax.set_title("Observed accuracy by support stratum (corpus A, 95% CI)",
             fontsize=10.5, pad=10)

fig.tight_layout()
fig.savefig("figures/fig1_coverage_curve.pdf", bbox_inches="tight")
fig.savefig("figures/fig1_coverage_curve.png", bbox_inches="tight")
print("wrote figures/fig1_coverage_curve.{pdf,png}")
