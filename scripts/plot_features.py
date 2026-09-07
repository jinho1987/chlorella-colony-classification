import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV = os.path.join(ROOT, "colony_features.csv")
OUT_FIG = os.path.join(ROOT, "strain_feature_comparison.png")

STRAINS = ["HS2", "ALE", "HMF", "HS19"]  # HS2 = mother strain first
COLORS = {"HS2": "#4C72B0", "ALE": "#DD8452", "HMF": "#55A868", "HS19": "#C44E52"}
FEATURES = [
    ("equiv_diameter_px", "Colony diameter (px)"),
    ("circularity", "Circularity"),
    ("mean_hue", "Mean hue (0-179)"),
    ("mean_sat", "Mean saturation"),
    ("mean_val", "Mean value (brightness)"),
]

rows = list(csv.DictReader(open(FEAT_CSV, encoding="utf-8")))
data = defaultdict(lambda: defaultdict(list))
for r in rows:
    for key, _ in FEATURES:
        data[key][r["strain"]].append(float(r[key]))

fig, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 4.5))
print(f"{'feature':22s} {'Kruskal-Wallis H':>18s} {'p-value':>10s}   n per strain")
for ax, (key, label) in zip(axes, FEATURES):
    groups = [data[key][s] for s in STRAINS]
    box = ax.boxplot(groups, labels=STRAINS, patch_artist=True, showmeans=True)
    for patch, s in zip(box["boxes"], STRAINS):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.6)
    for i, s in enumerate(STRAINS, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(data[key][s]))
        ax.scatter(np.full(len(data[key][s]), i) + jitter, data[key][s],
                   color="black", s=14, alpha=0.6, zorder=3)
    ax.set_title(label, fontsize=11)
    ax.tick_params(axis="x", rotation=0)

    h_stat, p_val = stats.kruskal(*groups)
    ax.text(0.5, -0.18, f"Kruskal-Wallis p={p_val:.3f}", transform=ax.transAxes,
            ha="center", fontsize=9, color="#333333")
    n_str = ", ".join(f"{s}={len(data[key][s])}" for s in STRAINS)
    print(f"{key:22s} {h_stat:18.2f} {p_val:10.4f}   {n_str}")

fig.suptitle("Colony morphology / color by strain (all derived from mother strain HS2)",
             fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUT_FIG, dpi=150)
print(f"\nSaved figure to {OUT_FIG}")
