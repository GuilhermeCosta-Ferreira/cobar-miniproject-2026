# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_wind_velocity(wind: dict) -> float:
    """
    Get the average wind direction at x (perpendicular to head direction)
    """
    wind_vector = _get_vector(wind)
    wind_x = _project_x(wind_vector)

    return wind_x


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _get_vector(wind: dict) -> np.ndarray:
    left_velocity = wind['l']['qvel']
    right_velocity = wind['r']['qvel']

    return np.mean([left_velocity, right_velocity], axis=0)

def _project_x(wind_vector: np.ndarray) -> float:
    return wind_vector[0]
