"""The 2023 script, finished: find the green arrow and say which way it points.

origin/detect.py thresholded green and showed you the mask. It never found
an arrow or a direction. This does both, and takes the two lessons from the
depth sweep along:

  --gray-world        white-balance first (kills the orange-post hallucination)
  --range-m METRES    divide the water back out when you know the distance

    python -m depthsight.detect                       # webcam
    python -m depthsight.detect --image scene.png     # one picture, prints the verdict
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .water import gray_world, unsubmerge

LOWER = np.array([50, 100, 100])
UPPER = np.array([70, 255, 255])
MIN_AREA = 600  # px; smaller blobs are noise or far away


def find_arrow(img_bgr: np.ndarray):
    """Return (contour, 'left'|'right'|'up'|'down', tip_xy) or None."""
    mask = cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), LOWER, UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        # An arrow is solid but notched: solidity well below a rectangle's 1.0.
        if area < MIN_AREA or not 0.55 < area / max(hull_area, 1) < 0.92:
            continue
        if best is None or area > cv2.contourArea(best):
            best = c
    if best is None:
        return None
    # Fill the shape, then look along its long axis: the head is the end that
    # is wider across. (The farthest point from the centroid is NOT reliable:
    # the head's bulk drags the centroid forward, so tail corners can win.)
    filled = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.drawContours(filled, [best], -1, 255, -1)
    ys, xs = np.nonzero(filled)
    pts = np.column_stack([xs, ys]).astype(float)
    centre = pts.mean(axis=0)
    axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
    along = (pts - centre) @ axis
    across = np.abs((pts - centre) @ np.array([-axis[1], axis[0]]))
    if across[along > 0].max() < across[along < 0].max():
        axis, along = -axis, -along  # make +axis point at the wide end
    tip = pts[np.argmax(along)]
    dx, dy = axis
    direction = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
    return best, direction, (int(tip[0]), int(tip[1]))


def prepare(img, gray: bool, range_m: float | None):
    if range_m is not None:
        img = unsubmerge(img, range_m)
    if gray:
        img = gray_world(img)
    return img


def annotate(img, found):
    out = img.copy()
    if found:
        c, direction, tip = found
        cv2.drawContours(out, [c], -1, (255, 255, 255), 2)
        cv2.circle(out, tip, 7, (0, 0, 255), -1)
        cv2.putText(out, f"arrow -> {direction}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        cv2.putText(out, "no arrow", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image")
    ap.add_argument("--gray-world", action="store_true")
    ap.add_argument("--range-m", type=float)
    ap.add_argument("--out", help="write the annotated image here")
    a = ap.parse_args()

    if a.image:
        img = cv2.imread(a.image)
        found = find_arrow(prepare(img, a.gray_world, a.range_m))
        print(found[1] if found else "no arrow")
        if a.out:
            cv2.imwrite(a.out, annotate(img, found))
        return

    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("green-at-depth", annotate(frame, find_arrow(prepare(frame, a.gray_world, a.range_m))))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
