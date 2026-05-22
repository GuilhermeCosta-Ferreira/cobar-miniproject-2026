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
    max_turn_velocity: float,
    scary_height: float,
    gain: float,
    slow_down_rate: float,
    height_power: float = 2.0,
    central_power: float = 2.0,
    top_k: int | None = None,
) -> np.ndarray:
    if len(centroids) == 0:
        return np.array([0.0, 0.0])

    image_height, image_width = image.shape[:2]

    centroids_array = np.asarray(centroids, dtype=float)

    xs = centroids_array[:, 0]
    heights = centroids_array[:, 2]

    # Normalized horizontal offset:
    # -1 = far left, 0 = center, +1 = far right
    offsets = (xs - image_width / 2) / (image_width / 2)
    offsets = np.clip(offsets, -1.0, 1.0)

    # Centrality:
    # 1 = center, 0 = edge
    centrality = 1.0 - np.abs(offsets)
    centrality = np.clip(centrality, 0.0, 1.0)

    # Normalize height relative to scary_height
    height_score = heights / scary_height
    height_score = np.clip(height_score, 0.0, None)

    # Optional: keep only the strongest objects
    raw_weights = (height_score ** height_power) * (centrality ** central_power)

    if top_k is not None and len(raw_weights) > top_k:
        strongest_indices = np.argsort(raw_weights)[-top_k:]
        offsets = offsets[strongest_indices]
        height_score = height_score[strongest_indices]
        centrality = centrality[strongest_indices]
        raw_weights = raw_weights[strongest_indices]

    weight_sum = np.sum(raw_weights)

    if weight_sum <= 1e-9:
        return np.array([0.0, 0.0])

    weights = raw_weights / weight_sum

    # Weighted average obstacle position
    weighted_offset = np.sum(weights * offsets)

    # Threat is high when objects are tall and central
    threat = np.max(height_score * centrality)
    threat = np.clip(threat, 0.0, 1.0)

    # Slow down or reverse depending on threat
    forward_velocity = -current_forward_velocity * slow_down_rate * threat

    # Turn according to weighted obstacle position
    turning_velocity = max_turn_velocity * np.tanh(gain * weighted_offset)

    return np.asarray([forward_velocity, turning_velocity])
