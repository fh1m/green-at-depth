"""Underwater image formation, simplified.

    I_c(d) = J_c * exp(-k_c * d) + B_c * (1 - exp(-k_c * d))

J is the colour the object really has, B is the veiling light the water
scatters back toward the camera, d is the water path in metres and k_c is
the per-channel attenuation coefficient. Red dies first, blue survives.

Coefficients approximate the pure-water absorption table of Pope & Fry (1997)
at roughly the camera's channel peaks (R ~ 600 nm, G ~ 530 nm, B ~ 460 nm).
Real lakes and pools attenuate more than pure water. ``turbidity`` scales
the coefficients to stand in for that; it is a knob, not a measurement.
"""

from __future__ import annotations

import numpy as np

# Pope & Fry 1997, pure water absorption [1/m], in BGR order to match OpenCV.
PURE_WATER_K_BGR = np.array([0.0094, 0.0447, 0.2224])

# Veiling light: the blue-green glow of open water (BGR, 0..1).
VEIL_BGR = np.array([0.55, 0.42, 0.08])


def submerge(img_bgr: np.ndarray, depth_m: float, turbidity: float = 4.0,
             noise_dn: float = 2.0, seed: int = 0) -> np.ndarray:
    """Return what ``img_bgr`` looks like through ``depth_m`` metres of water,
    as an 8-bit camera with ``noise_dn`` counts of read noise would record it."""
    j = img_bgr.astype(np.float64) / 255.0
    t = np.exp(-PURE_WATER_K_BGR * turbidity * depth_m)  # transmission per channel
    i = (j * t + VEIL_BGR * (1.0 - t)) * 255.0
    i += np.random.default_rng(seed).normal(0, noise_dn, i.shape)
    return np.clip(np.round(i), 0, 255).astype(np.uint8)


def unsubmerge(img_bgr: np.ndarray, depth_m: float, turbidity: float = 4.0) -> np.ndarray:
    """Invert the model when the range is known (DVL altitude, sonar, stereo).
    Divides by the transmission, so it also divides up the noise: the red
    channel's noise grows as exp(+k_r * d). That is where this method dies."""
    i = img_bgr.astype(np.float64) / 255.0
    t = np.exp(-PURE_WATER_K_BGR * turbidity * depth_m)
    j = (i - VEIL_BGR * (1.0 - t)) / t
    return np.clip(j * 255.0, 0, 255).astype(np.uint8)


def gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """Cheapest colour constancy there is: assume the scene averages to grey."""
    f = img_bgr.astype(np.float64)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    return np.clip(f * gain, 0, 255).astype(np.uint8)
