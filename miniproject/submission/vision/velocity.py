# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_velocity_vector(
    current_forward_velocity: float,
    image: np.ndarray,
    centroids: list[tuple[float, float, float]],
    scary_height: float,
) -> np.ndarray:
    if len(centroids) == 0:
        return np.array([0.0, 0.0])

    closest_centroid = max(centroids, key=lambda c: c[2])
    image_height, image_width = image.shape[:2]

    forward_velocity = 0.0
    if closest_centroid[2] >= scary_height:
        forward_velocity = current_forward_velocity * -1

    offset = (closest_centroid[0] - image_width / 2) / (image_width / 2)
    if offset == 0:
        return np.array([0.0, 0.0])

    turning_velocity = 5 if offset >= 0 else -5

    return np.asarray([forward_velocity, turning_velocity])


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
