"""
Train a strain classifier on the 53 single-colony crops and evaluate it
honestly: Leave-One-Plate-Out cross-validation (never train and test on
colonies from the same physical plate) plus a label-permutation test, so we
can tell whether any accuracy we see is a real signal or just what you'd
expect by chance / from the known imaging-batch confound.

Features are simple and interpretable (size, shape, color mean+spread) —
appropriate for n=53. A deep CNN would have vastly more parameters than
training examples here and would be even more prone to fitting the batch
confound; if this classical baseline can't beat chance under LOPO-CV, a CNN
would not fix that.
"""
import csv
import os
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "colony_manifest.csv")
OUT_DIR = ROOT

STRAINS = ["HS2", "ALE", "HMF", "HS19"]
N_PERMUTATIONS = 500
RNG = np.random.default_rng(0)


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def plate_group(strain, source_image):
    m = re.search(r"-(\d+)\.jpg$", source_image, flags=re.IGNORECASE)
    plate_num = m.group(1) if m else "0"
    return f"{strain}_plate{plate_num}"


def extract_features(crop_path):
    img = imread_unicode(crop_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 60, 10), (80, 255, 220))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5:
        return None
    colony_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(colony_mask, [c], -1, 255, thickness=cv2.FILLED)

    mean_std = cv2.meanStdDev(hsv, mask=colony_mask)
    mean_hsv = mean_std[0].flatten()
    std_hsv = mean_std[1].flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture_std = cv2.meanStdDev(gray, mask=colony_mask)[1].flatten()[0]

    equiv_diam = 2 * np.sqrt(area / np.pi)
    perim = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull_area if hull_area > 0 else 0

    return np.array([
        equiv_diam, circularity, solidity,
        mean_hsv[0], mean_hsv[1], mean_hsv[2],
        std_hsv[0], std_hsv[1], std_hsv[2],
        texture_std,
    ])


FEATURE_NAMES = [
    "equiv_diameter", "circularity", "solidity",
    "mean_hue", "mean_sat", "mean_val",
    "std_hue", "std_sat", "std_val",
    "texture_std",
]


def run_lopo(X, y, groups, model_fn):
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = model_fn()
        model.fit(X_train, y[train_idx])
        preds = model.predict(X_test)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(preds)
    return np.array(y_true_all), np.array(y_pred_all)


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    X_list, y_list, groups = [], [], []
    for r in rows:
        feat = extract_features(os.path.join(ROOT, r["crop_path"]))
        if feat is None:
            continue
        X_list.append(feat)
        y_list.append(r["strain"])
        groups.append(plate_group(r["strain"], r["source_image"]))

    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups)

    print(f"n colonies = {len(y)}, n plate-groups = {len(set(groups))}")
    print("group -> strain, n colonies:")
    grp_strain = defaultdict(lambda: None)
    grp_count = Counter()
    for g, s in zip(groups, y):
        grp_strain[g] = s
        grp_count[g] += 1
    for g in sorted(grp_count):
        print(f"  {g}: strain={grp_strain[g]}, n={grp_count[g]}")

    for model_name, model_fn in [
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0)),
        ("LogisticRegression", lambda: LogisticRegression(max_iter=2000, C=0.5)),
    ]:
        print(f"\n=== {model_name}, Leave-One-Plate-Out CV ===")
        y_true, y_pred = run_lopo(X, y, groups, model_fn)
        acc = accuracy_score(y_true, y_pred)
        print(f"Observed accuracy: {acc:.3f}  (chance for 4 balanced classes = 0.25)")
        cm = confusion_matrix(y_true, y_pred, labels=STRAINS)
        print("Confusion matrix (rows=true, cols=pred), order", STRAINS)
        print(cm)

        # permutation test: shuffle labels AT THE GROUP LEVEL, rerun LOPO, build null dist
        unique_groups = np.array(sorted(set(groups)))
        group_to_label = {g: grp_strain[g] for g in unique_groups}
        base_labels = np.array([group_to_label[g] for g in unique_groups])

        null_accs = []
        for _ in range(N_PERMUTATIONS):
            shuffled = RNG.permutation(base_labels)
            perm_map = dict(zip(unique_groups, shuffled))
            y_perm = np.array([perm_map[g] for g in groups])
            yt, yp = run_lopo(X, y_perm, groups, model_fn)
            null_accs.append(accuracy_score(yt, yp))
        null_accs = np.array(null_accs)
        p_value = (np.sum(null_accs >= acc) + 1) / (len(null_accs) + 1)
        print(f"Permutation null accuracy: mean={null_accs.mean():.3f} sd={null_accs.std():.3f}")
        print(f"Empirical p-value (observed >= null): {p_value:.4f}")

    # feature importance from a RF fit on ALL data, for interpretability only
    scaler = StandardScaler().fit(X)
    rf_full = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0)
    rf_full.fit(scaler.transform(X), y)
    print("\nFeature importances (RandomForest, fit on all data, for interpretation only):")
    for name, imp in sorted(zip(FEATURE_NAMES, rf_full.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:16s} {imp:.3f}")


if __name__ == "__main__":
    main()
