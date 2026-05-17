# ================================================================
# 0. Section: IMPORTS
# ================================================================
import cv2

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


def convert_to_hsv_fast(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        # For float32 OpenCV HSV:
        # H is [0, 360], S and V are [0, 1]
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # For uint8 OpenCV HSV:
    # H is [0, 179], S and V are [0, 255]
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
