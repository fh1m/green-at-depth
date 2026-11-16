"""Sink the test card metre by metre and score the 2023 HSV threshold.

    python -m depthsight.run          # writes assets/*.png and prints the table
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .scene import make_scene
from .water import gray_world, submerge, unsubmerge

# The exact range from origin/detect.py.
LOWER = np.array([50, 100, 100])
UPPER = np.array([70, 255, 255])

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEPTHS = np.arange(0, 20.5, 0.5)


def hsv_mask(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.inRange(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), LOWER, UPPER)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


STRATEGIES = {
    "2023 threshold, as-is": lambda img, d: img,
    "+ gray-world white balance": lambda img, d: gray_world(img),
    "+ invert the water (range known)": lambda img, d: unsubmerge(img, d),
}
STYLE = [("#2a6f4e", "-"), ("#c2571a", "--"), ("#1f4e8c", ":")]


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    scene, truth = make_scene()
    curves = {name: [iou(hsv_mask(f(submerge(scene, d), d)), truth) for d in DEPTHS]
              for name, f in STRATEGIES.items()}

    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for (name, ys), (color, ls) in zip(curves.items(), STYLE):
        ax.plot(DEPTHS, ys, color=color, lw=2.2, ls=ls, label=name)
    ax.set_xlabel("water path (m)")
    ax.set_ylabel("mask IoU vs ground truth")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "iou_vs_depth.png")
    plt.close(fig)

    # Filmstrip: camera view, then the mask each strategy keeps.
    rows = []
    for d in (0, 2, 5, 9, 14):
        under = submerge(scene, d)
        cols = [under] + [cv2.cvtColor(hsv_mask(f(under, d)), cv2.COLOR_GRAY2BGR)
                          for f in STRATEGIES.values()]
        row = np.hstack(cols)
        cv2.putText(row, f"{d} m", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(cv2.resize(row, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(ASSETS / "filmstrip.png"), np.vstack(rows))

    print("depth_m  " + "  ".join(f"{n[:24]:>24}" for n in curves))
    for i, d in enumerate(DEPTHS):
        if d in (0, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20):
            print(f"{d:7.1f}  " + "  ".join(f"{c[i]:24.3f}" for c in curves.values()))
    for name, ys in curves.items():
        dead = next((float(d) for d, v in zip(DEPTHS, ys) if v < 0.5), None)
        print(f"IoU < 0.5 first at {dead} m :: {name}")


if __name__ == "__main__":
    main()
