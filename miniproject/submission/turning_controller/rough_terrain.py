# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def damp_drives_for_rough_terrain(
    drives: list | np.ndarray,
    speed_scale: float = 0.75,
    turn_scale: float = 0.55,
    min_drive: float = 0.5,
    max_drive: float = 1.0,
) -> np.ndarray:
    """
    reduces:
    - forward speed
    - aggressive left/right drive differences
    """

    drives = np.asarray(drives, dtype=float).copy()

    mean_drive = np.mean(drives)
    turn_component = drives - mean_drive

    new_drives = mean_drive * speed_scale + turn_component * turn_scale

    return np.clip(new_drives, min_drive, max_drive)
