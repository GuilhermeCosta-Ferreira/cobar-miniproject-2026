# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np


# ================================================================
# 1. Section: Functions
# ================================================================
def get_signals_from_centroids(
    centroids: list[tuple[float, float]],
    image_shape: tuple[int, int, int] | np.ndarray,
    turn_gain: float = 0.8,
    min_signal: float = 0.5,
    max_signal: float = 1.5,
) -> np.ndarray:
    if len(centroids) == 0:
        return np.array([0.0, 0.0])

    image_height, image_width = image_shape[:2]

    # Use the average obstacle x-position.
    xs = np.array([cx for cx, _ in centroids])

    mean_x = xs.mean()

    # Horizontal offset:
    #   -1 = obstacle far left
    #    0 = obstacle centered
    #   +1 = obstacle far right
    offset = (mean_x - image_width / 2) / (image_width / 2)

    # Avoidance:
    # obstacle left  -> offset < 0 -> turn right -> left_signal > right_signal
    # obstacle right -> offset > 0 -> turn left  -> right_signal > left_signal
    turn = -turn_gain * offset

    left_signal = 0.0 + turn
    right_signal = 0.0 - turn

    return np.clip(
        np.array([left_signal, right_signal]),
        min_signal,
        max_signal,
    )
