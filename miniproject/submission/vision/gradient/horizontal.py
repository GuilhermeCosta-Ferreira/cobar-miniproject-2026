# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def horizontal_gradient(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    # Convert RGB image to grayscale
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    gray = gray.astype(np.float32)

    # Gradient along horizontal axis = axis 1
    grad_x = np.gradient(gray, axis=1)

    return np.abs(grad_x)
