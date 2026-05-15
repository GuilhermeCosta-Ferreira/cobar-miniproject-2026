# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from .convert import convert_to_hsv
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
    hsv_image = convert_to_hsv(image)
    hue, saturation, value = get_hsv_values(hsv_image)

    # Circular hue distance
    target = hue_to_degree(target_hue)
    tolerance_hue = hue_to_degree(tolerance_hue)

    # Saturation filter avoids meaningless hue values in gray/black/white areas
    diff = np.abs(hue - target)
    diff = np.minimum(diff, 1.0 - diff)
    mask = (diff <= tolerance_hue) & (saturation > min_saturation) & (value > min_value)

    return mask
