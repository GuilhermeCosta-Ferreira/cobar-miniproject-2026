# ================================================================
# 0. Section: IMPORTS
# ================================================================
import cv2

import numpy as np


# ================================================================
# 1. Section: Functions
# ================================================================
def get_hsv_mask(
    image: np.ndarray,
    target_hue: float,
    tolerance_hue: float,
    tolerance_value: float,
    tolerance_saturation: float,
    target_saturation: float,
    target_value: float,
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
    tol_val = tolerance_value * 0.01
    tol_sat = tolerance_saturation * 0.01

    s_low = int(np.clip(target_saturation - tol_sat, 0.0, 1.0) * 255)
    s_high = int(np.clip(target_saturation + tol_sat, 0.0, 1.0) * 255)

    v_low = int(np.clip(target_value - tol_val, 0.0, 1.0) * 255)
    v_high = int(np.clip(target_value + tol_val, 0.0, 1.0) * 255)

    lower = np.array(
        [
            max(0, target_h - tol_h),
            s_low,
            v_low,
        ],
        dtype=np.uint8,
    )

    upper = np.array(
        [
            min(179, target_h + tol_h),
            s_high,
            v_high,
        ],
        dtype=np.uint8,
    )

    return cv2.inRange(hsv, lower, upper).astype(bool)
