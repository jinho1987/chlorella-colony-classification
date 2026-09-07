import os
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP_DIR = os.path.join(ROOT, "cropped_colonies")
OUT_PATH = os.path.join(ROOT, "contact_sheet.jpg")

CELL = 140
COLS = 10


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)


strains = sorted(os.listdir(CROP_DIR))
rows_imgs = []
for strain in strains:
    files = sorted(os.listdir(os.path.join(CROP_DIR, strain)))[: COLS * 3]
    n_rows = (len(files) + COLS - 1) // COLS
    sheet = np.full((n_rows * CELL + 30, COLS * CELL, 3), 255, np.uint8)
    cv2.putText(sheet, f"{strain} (showing {len(files)} of {len(os.listdir(os.path.join(CROP_DIR, strain)))})",
                (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    for i, fname in enumerate(files):
        img = imread_unicode(os.path.join(CROP_DIR, strain, fname))
        if img is None:
            continue
        img = cv2.resize(img, (CELL - 6, CELL - 6))
        r, c = divmod(i, COLS)
        y0 = 30 + r * CELL + 3
        x0 = c * CELL + 3
        sheet[y0:y0 + CELL - 6, x0:x0 + CELL - 6] = img
    rows_imgs.append(sheet)

max_w = max(s.shape[1] for s in rows_imgs)
padded = []
for s in rows_imgs:
    if s.shape[1] < max_w:
        pad = np.full((s.shape[0], max_w - s.shape[1], 3), 255, np.uint8)
        s = np.hstack([s, pad])
    padded.append(s)
    padded.append(np.full((10, max_w, 3), 200, np.uint8))

full = np.vstack(padded)
imwrite_unicode(OUT_PATH, full)
print("wrote", OUT_PATH, full.shape)
