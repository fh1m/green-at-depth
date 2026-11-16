"""A RoboSub-flavoured test card: a green arrow, an orange gate post and
a sandy floor with some texture. Synthetic on purpose: the ground-truth
mask is known exactly, so the score measures the threshold, not the labeller."""

from __future__ import annotations

import cv2
import numpy as np

H, W = 360, 640
ARROW_BGR = (60, 200, 40)  # the green the 2023 script was tuned for
GATE_BGR = (20, 110, 235)


def make_scene(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    floor = np.empty((H, W, 3), np.float64)
    floor[:] = (150, 175, 190)  # sand, BGR
    floor += rng.normal(0, 9, (H, W, 1))  # grain
    img = np.clip(floor, 0, 255).astype(np.uint8)

    cv2.rectangle(img, (470, 40), (500, 330), GATE_BGR, -1)

    arrow = np.array([[120, 150], [300, 150], [300, 100], [400, 185],
                      [300, 270], [300, 220], [120, 220]], np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [arrow], 255)
    img[mask > 0] = ARROW_BGR
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, mask
