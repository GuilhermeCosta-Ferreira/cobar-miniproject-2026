# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def average_olfaction_signal(smooth_olfaction: np.ndarray) -> np.ndarray:
    """Assumes there is only one type of smell and averages over each antenna"""
    return np.average(
        smooth_olfaction[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
