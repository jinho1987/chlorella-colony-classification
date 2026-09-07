"""
The 4-way LOPO classifier came back at 0.019 accuracy (worse than chance).
Before concluding nothing separates at all, check every strain PAIR
individually (6 pairs) under the same honest LOPO protocol: maybe two
specific strains differ even though all four together don't.
"""
import csv
import itertools
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import train_classifier as tc

ROOT = tc.ROOT
N_PERM = 300


def main():
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
    rng = np.random.default_rng(1)

    print(f"{'pair':20s} {'n_A':>4s} {'n_B':>4s} {'n_plates':>9s} {'LOPO acc':>9s} {'null mean':>10s} {'p-value':>8s}")
    for a, b in itertools.combinations(tc.STRAINS, 2):
        mask = (y == a) | (y == b)
        Xs, ys, gs = X[mask], y[mask], groups[mask]
        n_plates = len(set(gs))
        if n_plates < 2:
            continue

        y_true, y_pred = tc.run_lopo(Xs, ys, gs, model_fn)
        acc = accuracy_score(y_true, y_pred)

        unique_groups = np.array(sorted(set(gs)))
        grp_label = {g: ys[gs == g][0] for g in unique_groups}
        base_labels = np.array([grp_label[g] for g in unique_groups])
        null_accs = []
        for _ in range(N_PERM):
            shuffled = rng.permutation(base_labels)
            perm_map = dict(zip(unique_groups, shuffled))
            y_perm = np.array([perm_map[g] for g in gs])
            if len(set(y_perm)) < 2:
                continue
            yt, yp = tc.run_lopo(Xs, y_perm, gs, model_fn)
            null_accs.append(accuracy_score(yt, yp))
        null_accs = np.array(null_accs)
        p_value = (np.sum(null_accs >= acc) + 1) / (len(null_accs) + 1)

        n_a, n_b = (y == a).sum(), (y == b).sum()
        print(f"{a+' vs '+b:20s} {n_a:4d} {n_b:4d} {n_plates:9d} {acc:9.3f} {null_accs.mean():10.3f} {p_value:8.4f}")


if __name__ == "__main__":
    main()
