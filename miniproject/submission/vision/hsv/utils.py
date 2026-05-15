# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_hsv_values(hsv_image: np.ndarray) -> tuple:
    hue = hsv_image[..., 0]
    saturation = hsv_image[..., 1]
    value = hsv_image[..., 2]

    return hue, saturation, value

def hue_to_degree(hue: float) -> float:
    return hue / 360.0
