"""Colours of the PCA maps: the global axes as red, green and blue, nudged by the local axes in OKLab.

OKLab (Ottosson, 2020) is perceptual: the same nudge looks as large on any colour. A cell keeps the colour
family of its global scores; how it differs from its surroundings shifts it a little lighter or darker
(``LIGHTNESS``) and turns its hue a little (``HUE``). The explorer Space does the same in JavaScript.
"""

import numpy as np

LIGHTNESS = 0.22  # OKLab lightness added at a local score of ±1
HUE = 45.0  # degrees of hue turned at a local score of ±1
_L_RANGE = (0.2, 0.97)  # keep nudged colours away from black and white

_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_FROM_LAB = np.linalg.inv(_TO_LAB)
_FROM_LMS = np.linalg.inv(_TO_LMS)


def _linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _gamma(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB rows in [0, 1] to OKLab rows (L, a, b)."""
    return np.cbrt(_linear(np.asarray(rgb, float)) @ _TO_LMS.T) @ _TO_LAB.T


def from_oklab(lab: np.ndarray) -> np.ndarray:
    """OKLab rows to sRGB rows, clipped to [0, 1]."""
    return np.clip(_gamma(((lab @ _FROM_LAB.T) ** 3) @ _FROM_LMS.T), 0, 1)


def nudge(rgb: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Each sRGB colour made lighter or darker by ``local[:, 0]`` and its hue turned by ``local[:, 1]``.

    Both local columns are in [-1, 1]: their ends give ``LIGHTNESS`` and ``HUE``.
    """
    lab = to_oklab(rgb)
    lightness = np.clip(lab[:, 0] + LIGHTNESS * local[:, 0], *_L_RANGE)
    angle = np.radians(HUE) * local[:, 1]
    a = lab[:, 1] * np.cos(angle) - lab[:, 2] * np.sin(angle)
    b = lab[:, 1] * np.sin(angle) + lab[:, 2] * np.cos(angle)
    return from_oklab(np.column_stack([lightness, a, b]))
