"""
Run this once pilot photos are shot, organized, and segmented
(run_pilot_segmentation.py). This is explain_strains.py's confound-ratio
check, but grouped by IMAGING SESSION instead of by physical plate --
because in the pilot design, session (not plate identity) is the thing
deliberately crossed with strain via interleaving, so session is the
variable to check "did we actually break the confound?" against.

Prints, per feature, the same between-strain vs. within-strain spread ratio
as the original analysis, plus the original dataset's numbers side-by-side
for direct comparison, plus a mechanical rig check on brightness
specifically (within-photo vs. within-session vs. between-session spread).
"""
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_classifier as tc

ROOT = tc.ROOT
PILOT_MANIFEST = os.path.join(ROOT, "colony_manifest_pilot.csv")
FEATURES = tc.FEATURE_NAMES

# from the original (confounded) dataset's README table -- for side-by-side comparison
ORIGINAL_RATIOS = {
    "equiv_diameter": 0.26, "circularity": 1.32, "solidity": 1.29,
    "mean_hue": 0.07, "mean_sat": 0.35, "mean_val": 0.75,
    "std_hue": 0.66, "std_sat": 0.33, "std_val": 0.64, "texture_std": 0.62,
}


def session_group(source_image):
    m = re.search(r"_s(\d+)_", source_image)
    return f"session{m.group(1)}" if m else "session_unknown"


def main():
    if not os.path.exists(PILOT_MANIFEST):
        print(f"No pilot manifest found at {PILOT_MANIFEST}.")
        print("Run run_pilot_segmentation.py on the pilot photos first.")
        sys.exit(1)

    rows = list(csv.DictReader(open(PILOT_MANIFEST, encoding="utf-8")))
    if not rows:
        print("Pilot manifest is empty -- no colonies were detected/accepted. Check QC overlays in qc_overlays_pilot/.")
        sys.exit(1)

    strains = sorted(set(r["strain"] for r in rows))
    feat_rows = []
    for r in rows:
        feat = tc.extract_features(os.path.join(ROOT, r["crop_path"]))
        if feat is None:
            continue
        feat_rows.append(dict(
            strain=r["strain"],
            session=session_group(r["source_image"]),
            source_image=r["source_image"],
            **dict(zip(FEATURES, feat)),
        ))

    print(f"n colonies (pilot) = {len(feat_rows)}, strains = {strains}, "
          f"sessions = {sorted(set(r['session'] for r in feat_rows))}\n")

    print(f"{'feature':16s} {'between-strain SD':>18s} {'within-strain SD':>18s} {'ratio':>7s} {'orig. ratio':>12s}")
    for f in FEATURES:
        session_means = defaultdict(list)
        for r in feat_rows:
            session_means[(r["strain"], r["session"])].append(r[f])
        cell_mean = {k: np.mean(v) for k, v in session_means.items()}
        cell_strain = {k: k[0] for k in cell_mean}

        strain_means = [np.mean([cell_mean[k] for k in cell_mean if cell_strain[k] == s]) for s in strains]
        between_strain_sd = np.std(strain_means, ddof=1) if len(strain_means) > 1 else float("nan")

        within_sds = []
        for s in strains:
            vals = [cell_mean[k] for k in cell_mean if cell_strain[k] == s]
            if len(vals) > 1:
                within_sds.append(np.std(vals, ddof=1))
        within_strain_sd = np.mean(within_sds) if within_sds else float("nan")

        ratio = between_strain_sd / within_strain_sd if within_strain_sd else float("nan")
        orig = ORIGINAL_RATIOS.get(f, float("nan"))
        flag = "  <-- improved" if ratio > orig and ratio > 0.9 else ""
        print(f"{f:16s} {between_strain_sd:18.2f} {within_strain_sd:18.2f} {ratio:7.2f} {orig:12.2f}{flag}")

    # mechanical rig check: brightness within-photo vs within-session vs between-session
    print("\nRig check (mean_val / brightness):")
    by_photo = defaultdict(list)
    for r in feat_rows:
        by_photo[r["source_image"]].append(r["mean_val"])
    within_photo_sds = [np.std(v, ddof=1) for v in by_photo.values() if len(v) > 1]
    print(f"  within-photo colony-to-colony SD (residual noise): {np.mean(within_photo_sds):.2f}"
          if within_photo_sds else "  (not enough colonies per photo to estimate)")

    by_session = defaultdict(list)
    for r in feat_rows:
        by_session[r["session"]].append(r["mean_val"])
    session_level_means = {k: np.mean(v) for k, v in by_session.items()}
    print(f"  between-session SD (residual lighting drift, want this SMALL): "
          f"{np.std(list(session_level_means.values()), ddof=1):.2f}"
          if len(session_level_means) > 1 else "  (need >1 session)")
    print("  (original dataset's between-PLATE-within-strain brightness SD was 10.24 -- "
          "compare against that as the pre-fix baseline)")


if __name__ == "__main__":
    main()
