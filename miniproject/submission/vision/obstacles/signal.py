# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np


# ================================================================
# 1. Section: Functions
# ================================================================
def get_signals_from_centroids(
    centroids: list[tuple[float, float, float]],
    image_shape: tuple[int, int, int] | np.ndarray,
    turn_gain: float = 7,
    min_signal: float = 0,
    max_signal: float = 1.5,
    expected_max_height: float | None = None,
) -> np.ndarray:
    if len(centroids) == 0:
        return np.array([0.0, 0.0])

    image_height, image_width = image_shape[:2]

    if expected_max_height is None:
        expected_max_height = float(image_height)

    scored_centroids = []

    for cx, cy, obs_height in centroids:
        # -1 = far left, 0 = center, +1 = far right
        offset = (cx - image_width / 2) / (image_width / 2)
        offset = np.clip(offset, -1.0, 1.0)

        # 1 when centered, 0 when at image edge
        centrality = 1.0 - abs(offset)

        # 0 = small/far obstacle, 1 = large/close obstacle
        height_weight = obs_height / expected_max_height
        height_weight = np.clip(height_weight, 0.0, 1.0)

        # Combined danger score
        centrality_importance = 0.3
        height_importance = 0.7

        risk = centrality_importance * centrality + height_importance * height_weight

        scored_centroids.append((risk, cx, cy, obs_height, offset))

    # Select the most dangerous obstacle, not just the tallest one
    risk, cx, cy, obs_height, offset = max(scored_centroids, key=lambda x: x[0])

    # print(f"SELECTED: cx={cx}, cy={cy}, height={obs_height}, risk={risk:.3f}")

    # Direction:
    # obstacle right -> offset > 0 -> turn left
    # obstacle left  -> offset < 0 -> turn right
    if offset >= 0:
        direction = -1.0
    else:
        direction = 1.0

    turn = turn_gain * risk * direction

    left_signal = turn
    right_signal = -turn

    return np.clip(
        np.array([left_signal, right_signal]),
        min_signal,
        max_signal,
    )
