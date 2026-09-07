"""
Which feature best characterizes each strain -- and does that
"characterization" survive the same batch-confound check we applied to
brightness earlier?

For each strain, one-vs-rest Cohen's d is computed per feature (how far that
strain's mean sits from the other three, in pooled-SD units). But a large
effect size alone doesn't mean "biology" here -- we already showed that
brightness looks hugely significant across strains yet varies almost as much
across a single strain's own plates. So for every feature this script also
computes a decisive number: the ratio of between-strain spread to
within-strain, between-plate spread (plate means only). If that ratio isn't
clearly > 1, the "explanation" is most likely session/lighting, not strain.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "colony_numeric_dataset.csv")
OUT_HEATMAP = os.path.join(ROOT, "strain_effect_size_heatmap.png")
OUT_TOPFEAT = os.path.join(ROOT, "top_feature_per_strain_by_plate.png")

STRAINS = ["HS2", "ALE", "HMF", "HS19"]
FEATURES = [
    "equiv_diameter", "circularity", "solidity",
    "mean_hue", "mean_sat", "mean_val",
    "std_hue", "std_sat", "std_val",
    "texture_std",
]


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_sd


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    data = {f: np.array([float(r[f]) for r in rows]) for f in FEATURES}
    strain = np.array([r["strain"] for r in rows])
    plate = np.array([r["plate_group"] for r in rows])

    # ---- 1) one-vs-rest Cohen's d, strain x feature ----
    effect = np.zeros((len(STRAINS), len(FEATURES)))
    for i, s in enumerate(STRAINS):
        for j, f in enumerate(FEATURES):
            in_grp = data[f][strain == s]
            out_grp = data[f][strain != s]
            effect[i, j] = cohens_d(in_grp, out_grp)

    print("Cohen's d (strain vs. rest), positive = strain is higher:\n")
    header = f"{'strain':6s} " + " ".join(f"{f[:10]:>10s}" for f in FEATURES)
    print(header)
    for i, s in enumerate(STRAINS):
        print(f"{s:6s} " + " ".join(f"{effect[i,j]:10.2f}" for j in range(len(FEATURES))))

    print("\nTop 2 distinguishing features per strain (by |Cohen's d|):")
    top_per_strain = {}
    for i, s in enumerate(STRAINS):
        order = np.argsort(-np.abs(effect[i]))[:2]
        top_per_strain[s] = [FEATURES[j] for j in order]
        for j in order:
            print(f"  {s}: {FEATURES[j]:16s} d={effect[i,j]:+.2f}")

    # ---- 2) batch-confound check for EVERY feature: between-strain spread
    #         vs within-strain, between-plate spread (using plate means) ----
    print("\nConfound check per feature: does between-strain spread exceed")
    print("within-strain (between-plate) spread? ratio > 1 favors real signal;")
    print("ratio <= 1 means plate-to-plate noise alone explains as much or more.\n")
    print(f"{'feature':16s} {'between-strain SD':>18s} {'within-strain SD':>18s} {'ratio':>7s}")
    confound_ratio = {}
    for f in FEATURES:
        plate_means = defaultdict(list)
        for r in rows:
            plate_means[r["plate_group"]].append(float(r[f]))
        plate_mean_val = {g: np.mean(v) for g, v in plate_means.items()}
        plate_strain = {g: g.split("_plate")[0] for g in plate_mean_val}

        strain_means = [np.mean([plate_mean_val[g] for g in plate_mean_val if plate_strain[g] == s])
                        for s in STRAINS]
        between_strain_sd = np.std(strain_means, ddof=1)

        within_strain_sds = []
        for s in STRAINS:
            vals = [plate_mean_val[g] for g in plate_mean_val if plate_strain[g] == s]
            if len(vals) > 1:
                within_strain_sds.append(np.std(vals, ddof=1))
        within_strain_sd = np.mean(within_strain_sds) if within_strain_sds else np.nan

        ratio = between_strain_sd / within_strain_sd if within_strain_sd else np.nan
        confound_ratio[f] = ratio
        print(f"{f:16s} {between_strain_sd:18.2f} {within_strain_sd:18.2f} {ratio:7.2f}")

    # ---- figure 1: effect size heatmap ----
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(effect, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=40, ha="right")
    ax.set_yticks(range(len(STRAINS))); ax.set_yticklabels(STRAINS)
    for i in range(len(STRAINS)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{effect[i,j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cohen's d (strain vs. rest)")
    ax.set_title("Which feature separates each strain from the other three?\n(before checking whether it's plate confound)")
    fig.tight_layout()
    fig.savefig(OUT_HEATMAP, dpi=150)
    print(f"\nSaved {OUT_HEATMAP}")

    # ---- figure 2: for each strain's #1 feature, plot by plate to show confound ----
    fig, axes = plt.subplots(1, len(STRAINS), figsize=(4.2 * len(STRAINS), 4))
    colors = {"HS2": "#4C72B0", "ALE": "#DD8452", "HMF": "#55A868", "HS19": "#C44E52"}
    for ax, s in zip(axes, STRAINS):
        top_feat = top_per_strain[s][0]
        vals_by_plate = defaultdict(list)
        for r in rows:
            if r["strain"] == s:
                vals_by_plate[r["plate_group"]].append(float(r[top_feat]))
        plates = sorted(vals_by_plate)
        for i, p in enumerate(plates):
            ys = vals_by_plate[p]
            xs = np.full(len(ys), i) + np.random.default_rng(0).normal(0, 0.05, len(ys))
            ax.scatter(xs, ys, color=colors[s], alpha=0.7)
            ax.scatter([i], [np.mean(ys)], color="black", marker="_", s=400, linewidths=2)
        ax.set_xticks(range(len(plates))); ax.set_xticklabels(plates, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{s}: top feature = {top_feat}\nd={effect[STRAINS.index(s), FEATURES.index(top_feat)]:+.2f}, "
                     f"confound ratio={confound_ratio[top_feat]:.2f}", fontsize=9)
        ax.set_ylabel(top_feat)
    fig.suptitle("Each strain's #1 distinguishing feature, broken down by its own physical plates\n"
                 "(if dots jump around this much within one strain, the feature isn't a reliable strain signature)")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT_TOPFEAT, dpi=150)
    print(f"Saved {OUT_TOPFEAT}")


if __name__ == "__main__":
    main()
