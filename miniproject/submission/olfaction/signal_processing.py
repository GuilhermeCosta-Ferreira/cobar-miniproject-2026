# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_smooth_olfaction(
    olfaction_smooth: np.ndarray | None,
    signal: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Applies a smoothing, when possible to olfactation"""
    if olfaction_smooth is None:
        olfaction_smooth = signal
    else:
        olfaction_smooth = (
            1 - alpha
        ) * olfaction_smooth + alpha * signal
    return olfaction_smooth

def get_average_signal(olfaction_smooth: np.ndarray) -> np.ndarray:
    """Assumes there is only one type of smell and averages over each antenna"""
    average = np.average(
        olfaction_smooth[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    return average


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
