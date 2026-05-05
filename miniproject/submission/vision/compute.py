# ================================================================
# 0. Section: IMPORTS
# ================================================================
import cv2
import numpy as np

from miniproject.simulation import MiniprojectSimulation



# ================================================================
# 1. Section: Functions
# ================================================================
def detect_objects(sim: MiniprojectSimulation, vision: np.ndarray) -> np.ndarray:
    """Detect objects in the fly's field of view using the raw vision data"""
    # Simple thresholding to detect objects (this is a placeholder, you can implement more sophisticated methods)
    gray = cv2.cvtColor(vision, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return mask