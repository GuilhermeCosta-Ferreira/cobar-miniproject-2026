# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from matplotlib.colors import rgb_to_hsv



# ================================================================
# 1. Section: Functions
# ================================================================
def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    # Convert to float RGB in [0, 1]
    if image.dtype == np.uint8:
        image_rgb = image.astype(np.float32) / 255.0
    else:
        image_rgb = image.astype(np.float32)

        # If image is accidentally in [0, 255]
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb / 255.0

    return rgb_to_hsv(image_rgb)
