import cv2
import numpy as np
import sys


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


path = sys.argv[1]
img = imread_unicode(path)  # BGR
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, w = img.shape[:2]
print("image size", w, h)

points = {
    "bg_top_left": (30, 30),
    "bg_top_right": (w - 30, 30),
    "bg_bottom": (w // 2, h - 20),
}

for name, (x, y) in points.items():
    b, g, r = img[y, x].tolist()
    hh, ss, vv = hsv[y, x].tolist()
    print(f"{name}: BGR=({b},{g},{r}) HSV=({hh},{ss},{vv})")

# grid sample center region to find dish/agar vs colony vs background stats
print("\n--- grid sample across image (every 100px) ---")
for y in range(0, h, 150):
    row = []
    for x in range(0, w, 150):
        hh, ss, vv = hsv[y, x].tolist()
        row.append(f"({hh:3d},{ss:3d},{vv:3d})")
    print(y, " ".join(row))
