"""
Detect well-isolated single Chlorella colonies on streak-plate photos and crop them out.

Pipeline per plate photo:
  1. Find the petri dish (distinguish dish interior from the pale blue-gray
     background) and restrict analysis to inside it, eroded slightly to avoid
     rim glare.
  2. Threshold for saturated green pixels (colony/streak biomass) in HSV space.
  3. Connected-component analysis on the green mask -> candidate blobs.
  4. Keep only blobs that look like a single isolated round colony:
       - plausible area (not dust, not a merged clump)
       - high circularity and solidity (rejects streak fragments and most
         handwritten ink strokes, which are elongated or hollow)
       - roughly square bounding box (extra guard against pen strokes/digits)
       - isolated: no other green blob within ISOLATION_PAD px (rejects
         colonies touching a streak or a neighboring colony)
  5. Crop an accepted colony (padded square) from the original color image.
  6. Write a QC overlay image per plate (green box = kept, red box = rejected,
     labeled with the rejection reason) so a human can sanity-check the
     automatic calls before anything is used for training.

This is deliberately conservative: on a dataset this small and this visually
uniform across strains, it is much safer to under-crop (miss some real
colonies) than to let handwriting or streak fragments leak into the training
set as mislabeled "colonies".
"""

import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np

# ---- paths -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # colony_pics/
SRC_DIR = os.path.join(ROOT, "Colony picture (1)", "Colony picture")
CROP_DIR = os.path.join(ROOT, "cropped_colonies")
QC_DIR = os.path.join(ROOT, "qc_overlays")
MANIFEST_PATH = os.path.join(ROOT, "colony_manifest.csv")

# ---- tuned thresholds (calibrated on this dataset's ~2000x1450 photos) -----
BG_HUE = (90, 120)      # pale blue-gray background, outside the dish
BG_SAT_MAX = 60
BG_VAL_MIN = 195

GREEN_HUE = (25, 75)
GREEN_SAT_MIN = 70
GREEN_VAL_MIN = 15
GREEN_VAL_MAX = 210

DISH_ERODE_PX = 18       # keep away from rim glare/reflection
MIN_AREA = 25
MAX_AREA = 3500
MIN_CIRCULARITY = 0.55
MIN_SOLIDITY = 0.85
MAX_ASPECT = 1.8         # bbox w/h or h/w, guards against pen strokes
MIN_DIAM = 6             # min(w,h) px; rejects sub-pixel noise/JPEG specks
GREEN_MARGIN = 6         # loose sanity floor: real colonies measured at dG-R~10-17,
                         # dG-B~12-35 on this dataset; this just excludes neutral
                         # gray/black ink, not a primary discriminator
CROP_MARGIN_PX = 10     # fixed margin added around the colony's own bbox
MIN_CROP_SIZE = 36      # floor so tiny colonies still get a usable crop
CROP_CONFLICT_PAD = 2    # px tolerance when checking if another blob falls in the crop


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    area: float
    circularity: float
    solidity: float
    accepted: bool = False
    reason: str = ""
    crop_box: tuple = None  # (x0, y0, x1, y1) once computed


def find_dish_roi(hsv, shape):
    h, w = shape[:2]
    bg_mask = cv2.inRange(
        hsv, (BG_HUE[0], 0, BG_VAL_MIN), (BG_HUE[1], BG_SAT_MAX, 255)
    )
    dish_mask = cv2.bitwise_not(bg_mask)
    dish_mask = cv2.morphologyEx(
        dish_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)
    )
    dish_mask = cv2.morphologyEx(
        dish_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)
    )

    contours, _ = cv2.findContours(
        dish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        # fall back: whole image
        return np.ones((h, w), np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)
    roi = np.zeros((h, w), np.uint8)
    cv2.drawContours(roi, [largest], -1, 255, thickness=cv2.FILLED)
    roi = cv2.erode(roi, np.ones((DISH_ERODE_PX, DISH_ERODE_PX), np.uint8))
    return roi


def rects_overlap_with_pad(a, b, pad):
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    ax1 -= pad; ay1 -= pad; ax2 += pad; ay2 += pad
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def process_image(path, out_crop_dir, out_qc_path):
    img = imread_unicode(path)
    if img is None:
        print(f"  !! could not read {path}")
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    dish_roi = find_dish_roi(hsv, img.shape)

    green_mask = cv2.inRange(
        hsv,
        (GREEN_HUE[0], GREEN_SAT_MIN, GREEN_VAL_MIN),
        (GREEN_HUE[1], 255, GREEN_VAL_MAX),
    )
    green_mask = cv2.bitwise_and(green_mask, green_mask, mask=dish_roi)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    all_bboxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        all_bboxes.append((x, y, bw, bh))
        perim = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        aspect = max(bw, bh) / max(1, min(bw, bh))

        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        mean_b, mean_g, mean_r, _ = cv2.mean(img, mask=mask)

        cand = Candidate(x, y, bw, bh, cx, cy, area, circularity, solidity)

        if area < MIN_AREA:
            cand.reason = "too small"
        elif min(bw, bh) < MIN_DIAM:
            cand.reason = "too thin/small"
        elif area > MAX_AREA:
            cand.reason = "too large / merged"
        elif circularity < MIN_CIRCULARITY:
            cand.reason = f"not round (circ={circularity:.2f})"
        elif solidity < MIN_SOLIDITY:
            cand.reason = f"irregular shape (sol={solidity:.2f})"
        elif aspect > MAX_ASPECT:
            cand.reason = f"elongated (aspect={aspect:.2f})"
        elif (mean_g - mean_r) < GREEN_MARGIN or (mean_g - mean_b) < GREEN_MARGIN:
            cand.reason = f"not green enough (likely ink, dG={mean_g-mean_r:.0f}/{mean_g-mean_b:.0f})"
        elif x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            cand.reason = "touches image edge"
        else:
            cand.accepted = True  # crop-frame isolation checked in second pass
        candidates.append(cand)

    # crop-frame isolation pass: compute the exact crop box, then reject if any
    # OTHER detected blob (colony, streak fragment, or ink) falls inside it.
    # This guarantees a saved crop shows exactly one object, matching what a
    # human reviewer will actually see in the file.
    for cand in candidates:
        if not cand.accepted:
            continue
        side = max(cand.w, cand.h) + 2 * CROP_MARGIN_PX
        side = max(side, MIN_CROP_SIZE)
        x0 = int(cand.cx - side / 2); y0 = int(cand.cy - side / 2)
        x1 = int(cand.cx + side / 2); y1 = int(cand.cy + side / 2)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        cand.crop_box = (x0, y0, x1, y1)
        crop_rect = (x0, y0, x1 - x0, y1 - y0)

        my_box = (cand.x, cand.y, cand.w, cand.h)
        for other_box in all_bboxes:
            if other_box == my_box:
                continue
            if rects_overlap_with_pad(crop_rect, other_box, -CROP_CONFLICT_PAD):
                cand.accepted = False
                cand.reason = "other object inside crop frame"
                break

    # crop accepted colonies + build QC overlay
    overlay = img.copy()
    os.makedirs(out_crop_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    accepted_records = []
    colony_idx = 0

    for cand in candidates:
        color = (0, 200, 0) if cand.accepted else (0, 0, 220)
        cv2.rectangle(
            overlay, (cand.x, cand.y), (cand.x + cand.w, cand.y + cand.h), color, 2
        )
        if not cand.accepted and cand.reason:
            cv2.putText(
                overlay, cand.reason, (cand.x, max(0, cand.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        if cand.accepted:
            x0, y0, x1, y1 = cand.crop_box
            crop = img[y0:y1, x0:x1]
            colony_idx += 1
            crop_name = f"{base}__colony{colony_idx:02d}.jpg"
            crop_path = os.path.join(out_crop_dir, crop_name)
            imwrite_unicode(crop_path, crop)
            accepted_records.append(
                dict(
                    crop_path=os.path.relpath(crop_path, ROOT),
                    source_image=os.path.relpath(path, ROOT),
                    x=cand.x, y=cand.y, w=cand.w, h=cand.h,
                    area=round(cand.area, 1),
                    circularity=round(cand.circularity, 3),
                    solidity=round(cand.solidity, 3),
                )
            )

    cv2.putText(
        overlay,
        f"kept {colony_idx} / {len(candidates)} candidates",
        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2, cv2.LINE_AA,
    )
    os.makedirs(os.path.dirname(out_qc_path), exist_ok=True)
    imwrite_unicode(out_qc_path, overlay)

    return accepted_records


def main():
    strains = sorted(
        d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))
    )
    print(f"Strains found: {strains}")

    manifest_rows = []
    for strain in strains:
        strain_dir = os.path.join(SRC_DIR, strain)
        images = sorted(
            f for f in os.listdir(strain_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        strain_crop_dir = os.path.join(CROP_DIR, strain)
        total_kept = 0
        for fname in images:
            path = os.path.join(strain_dir, fname)
            qc_path = os.path.join(QC_DIR, strain, os.path.splitext(fname)[0] + "_overlay.jpg")
            records = process_image(path, strain_crop_dir, qc_path)
            for r in records:
                r["strain"] = strain
            manifest_rows.extend(records)
            total_kept += len(records)
            print(f"  {strain}/{fname}: kept {len(records)} colonies")
        print(f"{strain}: {total_kept} colonies total from {len(images)} photos")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strain", "crop_path", "source_image",
                "x", "y", "w", "h", "area", "circularity", "solidity",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTotal colonies extracted: {len(manifest_rows)}")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"QC overlays written to {QC_DIR}")


if __name__ == "__main__":
    main()
