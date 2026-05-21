# ================================================================
# 0. Section: IMPORTS
# ================================================================
import cv2

import numpy as np

from ...world import timer, print_timings
from .convert import convert_to_hsv_fast
from .utils import hue_to_degree, get_hsv_values


# ================================================================
# 1. Section: Functions
# ================================================================
def get_hsv_mask(
    image: np.ndarray,
    target_hue: float,
    tolerance_hue: float,
    min_saturation: float,
    min_value: float,
) -> np.ndarray:
    timings: dict[str, float] = {}

    with timer("1.1_convert_to_hsv", timings):
        hsv_image = convert_to_hsv_fast(image)
    with timer("1.2_get_hsv_values", timings):
        hue, saturation, value = get_hsv_values(hsv_image)

    # Circular hue distance
    with timer("2_convert_to_degree", timings):
        target = hue_to_degree(target_hue)
        tolerance_hue = hue_to_degree(tolerance_hue)

    # Saturation filter avoids meaningless hue values in gray/black/white areas
    with timer("3_get_mask", timings):
        mask = get_mask_fast(
            hue, saturation, value, target, tolerance_hue, min_saturation, min_value
        )

    print_timings(timings)

    return mask


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_mask_fast(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    target: float,
    tolerance_hue: float,
    min_saturation: float,
    min_value: float,
) -> np.ndarray:
    lower_hue = target - tolerance_hue
    upper_hue = target + tolerance_hue

    mask = np.empty(hue.shape, dtype=bool)
    tmp = np.empty(hue.shape, dtype=bool)

    np.greater_equal(hue, lower_hue, out=mask)

    np.less_equal(hue, upper_hue, out=tmp)
    np.logical_and(mask, tmp, out=mask)

    np.greater(saturation, min_saturation, out=tmp)
    np.logical_and(mask, tmp, out=mask)

    np.greater(value, min_value, out=tmp)
    np.logical_and(mask, tmp, out=mask)

    return mask


def get_hsv_mask_fast(
    image: np.ndarray,
    target_hue: float,
    tolerance_hue: float,
    min_saturation: float,
    min_value: float,
) -> np.ndarray:
    """
    Fast HSV mask for uint8 RGB images.

    Assumes:
        image dtype is uint8
        image range is [0, 255]
        image channel order is RGB
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # OpenCV uint8 HSV:
    # H is [0, 179], S is [0, 255], V is [0, 255]
    target_h = int(target_hue / 2)
    tol_h = max(1, int(tolerance_hue / 2))

    lower = np.array(
        [
            max(0, target_h - tol_h),
            int(min_saturation * 255),
            int(min_value * 255),
        ],
        dtype=np.uint8,
    )

    upper = np.array(
        [
            min(179, target_h + tol_h),
            255,
            255,
        ],
        dtype=np.uint8,
    )

    return cv2.inRange(hsv, lower, upper).astype(bool)
