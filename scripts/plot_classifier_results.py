"""Re-run the RandomForest LOPO + permutation test from train_classifier.py
and save a figure: observed accuracy vs. the permutation null distribution,
plus the confusion matrix. Kept separate from train_classifier.py so the
console-report script stays simple to read.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_classifier as tc

ROOT = tc.ROOT
OUT_FIG = os.path.join(ROOT, "classifier_lopo_results.png")


def main():
    import csv
    rows = list(csv.DictReader(open(tc.MANIFEST, encoding="utf-8")))
    X_list, y_list, groups = [], [], []
    for r in rows:
        feat = tc.extract_features(os.path.join(ROOT, r["crop_path"]))
        if feat is None:
            continue
        X_list.append(feat)
        y_list.append(r["strain"])
        groups.append(tc.plate_group(r["strain"], r["source_image"]))
    X, y, groups = np.array(X_list), np.array(y_list), np.array(groups)

    model_fn = lambda: RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0)
    y_true, y_pred = tc.run_lopo(X, y, groups, model_fn)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=tc.STRAINS)

    unique_groups = np.array(sorted(set(groups)))
    grp_strain = {g: y[groups == g][0] for g in unique_groups}
    base_labels = np.array([grp_strain[g] for g in unique_groups])
    null_accs = []
    for _ in range(tc.N_PERMUTATIONS):
        shuffled = tc.RNG.permutation(base_labels)
        perm_map = dict(zip(unique_groups, shuffled))
        y_perm = np.array([perm_map[g] for g in groups])
        yt, yp = tc.run_lopo(X, y_perm, groups, model_fn)
        null_accs.append(accuracy_score(yt, yp))
    null_accs = np.array(null_accs)
    p_value = (np.sum(null_accs >= acc) + 1) / (len(null_accs) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(null_accs, bins=20, color="#9098A0", alpha=0.8, label="permuted labels\n(null distribution)")
    ax.axvline(acc, color="#C44E52", linewidth=2.5, label=f"observed accuracy = {acc:.3f}")
    ax.axvline(0.25, color="#333333", linestyle="--", linewidth=1, label="balanced chance (0.25)")
    ax.set_xlabel("Leave-one-plate-out accuracy")
    ax.set_ylabel("Permutation count")
    ax.set_title(f"Real strain labels vs. shuffled labels\np = {p_value:.3f} (observed ≥ null)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(tc.STRAINS))); ax.set_xticklabels(tc.STRAINS)
    ax.set_yticks(range(len(tc.STRAINS))); ax.set_yticklabels(tc.STRAINS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (LOPO-CV)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Colony-image classifier: RandomForest, Leave-One-Plate-Out CV (n=53 colonies, 10 plates)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Saved {OUT_FIG}")
    print(f"Observed acc={acc:.3f}, null mean={null_accs.mean():.3f}, p={p_value:.4f}")


if __name__ == "__main__":
    main()
