# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_velocity_vector(
    lat_olfaction: np.ndarray,
    forward_velocity: float,
    max_turn_velocity: float,
    min_forward_velocity: float,
) -> np.ndarray:
    """Builds a vector with forward and yaw rate"""
    mean_odor = lat_olfaction.mean()
    diff_odor = lat_olfaction[0] - lat_olfaction[1]
    odor_bias = diff_odor / mean_odor if mean_odor != 0 else 0

    rotational_velocity = 0.0
    if odor_bias != 0:
        turn_command = np.tanh(3 * odor_bias)
        rotational_velocity = max_turn_velocity * turn_command

        # Slows down to avoid sharp turns
        turn_strength = abs(turn_command)
        forward_velocity = forward_velocity * (1.0 - 0.5 * turn_strength)
        forward_velocity = max(forward_velocity, min_forward_velocity)

    velocity = np.array([forward_velocity, rotational_velocity])
    return velocity
