import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP_DIR = os.path.join(ROOT, "cropped_colonies")


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)


strain = sys.argv[1]
CELL = 160
COLS = 6
files = sorted(os.listdir(os.path.join(CROP_DIR, strain)))
n_rows = (len(files) + COLS - 1) // COLS
sheet = np.full((n_rows * (CELL + 18), COLS * CELL, 3), 255, np.uint8)
for i, fname in enumerate(files):
    img = imread_unicode(os.path.join(CROP_DIR, strain, fname))
    img = cv2.resize(img, (CELL - 6, CELL - 6))
    r, c = divmod(i, COLS)
    y0 = r * (CELL + 18)
    x0 = c * CELL
    sheet[y0:y0 + CELL - 6, x0:x0 + CELL - 6] = img
    cv2.putText(sheet, f"{i}", (x0 + 2, y0 + CELL + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

out = os.path.join(ROOT, f"labeled_sheet_{strain}.jpg")
imwrite_unicode(out, sheet)
print("wrote", out, "n=", len(files))
