# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_smooth_vision(
    olfaction_smooth: np.ndarray | None,
    signal: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Applies a smoothing, when possible to vision"""
    if olfaction_smooth is None:
        olfaction_smooth = signal
    else:
        olfaction_smooth = (1 - alpha) * olfaction_smooth + alpha * signal
    return olfaction_smooth
