# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np


# ================================================================
# 1. Section: Functions
# ================================================================
def get_velocity_vector(
    image: np.ndarray,
    centroids: list[tuple[float, float, float]],
    current_forward_velocity: float,
    gain: float,
    slow_down_rate: float,
    h_far = 0.0,
    h_stop = 0.95, #0.45
) -> np.ndarray:
    if len(centroids) == 0:
        return np.array([0.0, 0.0])
    biggest = max(centroids, key=lambda c: c[2])
    image_height, image_width = image.shape[:2]

    centroid_x = biggest[0]
    centroid_h = biggest[2]

    centrality = (centroid_x - image_width / 2) / (image_width / 2)
    height_ratio = centroid_h / image_height

    risk = np.clip((height_ratio - h_far) / (h_stop - h_far), 0.0, 1.0)
    risk = risk**2

    vf = -(current_forward_velocity * slow_down_rate) * risk #0.0

    if abs(centrality) < 0.1 and risk > 0.5:
        vt = gain * risk * 1
    else:
        vt = gain * risk * np.sign(centrality)

    return np.array([vf, vt])
