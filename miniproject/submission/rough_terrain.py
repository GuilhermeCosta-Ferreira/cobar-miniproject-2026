# rough_terrain.py

import numpy as np


def damp_drives_for_rough_terrain(
    drives,
    speed_scale=0.75,
    turn_scale=0.55,
    min_drive=0.5,
    max_drive=1.0,
):
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