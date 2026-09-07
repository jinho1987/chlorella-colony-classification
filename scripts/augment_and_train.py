"""
Expand the 53 real colony crops with image augmentation and re-run the same
honest LOPO-CV + permutation-test evaluation, to see whether augmentation
changes the conclusion from train_classifier.py.

What augmentation CAN do here: colony orientation on the plate is arbitrary,
so rotation/flip augmentation is free extra information (a real invariance,
not manufactured data). Brightness/contrast/saturation jitter is also
well-motivated for a different reason: we showed brightness is a plate/
lighting artifact, not biology, so training the model on deliberately
relit copies of each colony should make it LESS able to lean on absolute
brightness -- directly targeting the confound we diagnosed.

What augmentation CANNOT do: manufacture new physical plates or new imaging
sessions. Every augmented copy of a colony is still tied to the same plate
it came from (same tag used for LOPO grouping below), so this cannot leak
information across the train/test boundary, but it also cannot invent the
missing between-session diversity that would let us tell "strain" apart
from "which day this was shot".
"""
import csv
import os

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import train_classifier as tc

ROOT = tc.ROOT
N_AUG_PER_IMAGE = 9   # + the original = 10x effective dataset size
N_PERMUTATIONS = 200  # fewer than baseline script: each perm now retrains on ~10x the rows
RNG = np.random.default_rng(42)


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def augment_image(img, rng):
    h, w = img.shape[:2]

    # 1) random rotation about the crop center (colony orientation is arbitrary)
    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

    # 2) random flips
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        out = cv2.flip(out, 0)

    # 3) brightness / contrast / saturation jitter in HSV space -- this is
    #    the part specifically meant to fight the lighting confound.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_scale = rng.uniform(0.6, 1.5)     # simulate a differently-lit photo
    v_offset = rng.uniform(-15, 15)
    s_scale = rng.uniform(0.8, 1.2)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_scale + v_offset, 0, 255)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4) mild Gaussian noise (sensor/JPEG-artifact stand-in)
    noise = rng.normal(0, 4, out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def build_augmented_dataset():
    rows = list(csv.DictReader(open(tc.MANIFEST, encoding="utf-8")))
    X_list, y_list, groups = [], [], []
    n_ok = 0
    for r in rows:
        img = imread_unicode(os.path.join(ROOT, r["crop_path"]))
        if img is None:
            continue
        grp = tc.plate_group(r["strain"], r["source_image"])

        feat = tc.extract_features_from_image(img)
        if feat is not None:
            X_list.append(feat); y_list.append(r["strain"]); groups.append(grp)
            n_ok += 1

        for _ in range(N_AUG_PER_IMAGE):
            aug_img = augment_image(img, RNG)
            feat = tc.extract_features_from_image(aug_img)
            if feat is not None:
                X_list.append(feat); y_list.append(r["strain"]); groups.append(grp)

    print(f"Base crops usable: {n_ok} / {len(rows)}")
    return np.array(X_list), np.array(y_list), np.array(groups)


def evaluate(X, y, groups, model_fn, label):
    print(f"\n=== {label}: Leave-One-Plate-Out CV, n={len(y)} rows over {len(set(groups))} plates ===")
    y_true, y_pred = tc.run_lopo(X, y, groups, model_fn)
    acc = accuracy_score(y_true, y_pred)
    print(f"Observed accuracy: {acc:.3f}")
    cm = confusion_matrix(y_true, y_pred, labels=tc.STRAINS)
    print("Confusion matrix", tc.STRAINS)
    print(cm)

    unique_groups = np.array(sorted(set(groups)))
    grp_label = {g: y[groups == g][0] for g in unique_groups}
    base_labels = np.array([grp_label[g] for g in unique_groups])
    null_accs = []
    for _ in range(N_PERMUTATIONS):
        shuffled = RNG.permutation(base_labels)
        perm_map = dict(zip(unique_groups, shuffled))
        y_perm = np.array([perm_map[g] for g in groups])
        yt, yp = tc.run_lopo(X, y_perm, groups, model_fn)
        null_accs.append(accuracy_score(yt, yp))
    null_accs = np.array(null_accs)
    p_value = (np.sum(null_accs >= acc) + 1) / (len(null_accs) + 1)
    print(f"Permutation null: mean={null_accs.mean():.3f} sd={null_accs.std():.3f}  p={p_value:.4f}")
    return acc, p_value


def main():
    print("Building augmented dataset...")
    X_aug, y_aug, groups_aug = build_augmented_dataset()

    model_fn = lambda: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0)
    evaluate(X_aug, y_aug, groups_aug, model_fn, "RandomForest + augmentation (x10 data)")

    model_fn2 = lambda: LogisticRegression(max_iter=2000, C=0.5)
    evaluate(X_aug, y_aug, groups_aug, model_fn2, "LogisticRegression + augmentation (x10 data)")


if __name__ == "__main__":
    main()
