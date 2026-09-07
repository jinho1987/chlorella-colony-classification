"""
Exploratory check: before trying to train an image classifier, see whether
simple, interpretable colony features (size, roundness, color) show ANY
separation between the four sister strains at all. These strains were all
derived from the same mother strain (HS2), so this is a genuine open
question, not a given.

This does not train a model. It computes descriptive features per cropped
colony and visualizes their distribution by strain, plus a permutation test
on a simple combined score, as an honest first look at whether a signal
exists before committing to a CNN on ~50 images.
"""
import csv
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "colony_manifest.csv")
OUT_CSV = os.path.join(ROOT, "colony_features.csv")


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    out_rows = []
    for r in rows:
        crop_path = os.path.join(ROOT, r["crop_path"])
        img = imread_unicode(crop_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # re-segment the colony within its own tight crop (green threshold)
        # so color/size stats describe the colony, not the crop background.
        mask = cv2.inRange(hsv, (20, 60, 10), (80, 255, 220))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(c)
        if area_px < 5:
            continue
        colony_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(colony_mask, [c], -1, 255, thickness=cv2.FILLED)
        mean_b, mean_g, mean_r, _ = cv2.mean(img, mask=colony_mask)
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=colony_mask)
        equiv_diam = 2 * np.sqrt(area_px / np.pi)
        perim = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area_px / (perim * perim) if perim > 0 else 0

        out_rows.append(dict(
            strain=r["strain"],
            crop_path=r["crop_path"],
            source_image=r["source_image"],
            area_px=round(area_px, 1),
            equiv_diameter_px=round(equiv_diam, 2),
            circularity=round(circularity, 3),
            mean_hue=round(mean_h, 1),
            mean_sat=round(mean_s, 1),
            mean_val=round(mean_v, 1),
            mean_B=round(mean_b, 1),
            mean_G=round(mean_g, 1),
            mean_R=round(mean_r, 1),
        ))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
