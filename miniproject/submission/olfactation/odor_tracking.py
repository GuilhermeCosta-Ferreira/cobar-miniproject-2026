# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def odor_intensity_to_control_signal(
    lateralized_odor: np.ndarray,
    attractive_gain: int = -500,
) -> np.ndarray:
    """Convert odor sensor readings to a turning control signal.
    Assumes there is only one odor

    Parameters
    ----------
    odor_intensities : np.ndarray
        Average odor intensities from the 2x2 sensors, shape ``(2, ,)``.
    attractive_gain : float
        Gain applied to the attractive odor dimension.

    Returns
    -------
    np.ndarray
        Control signal of shape ``(2,)`` for left and right descending drive.
    """
    odor_bias = (
        attractive_gain
        * (lateralized_odor[0] - lateralized_odor[1])
        / lateralized_odor.mean()
        if lateralized_odor.mean() != 0
        else 0
    )

    effective_bias = odor_bias
    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
    assert np.sign(effective_bias_norm) == np.sign(effective_bias)

    control_signal = np.ones(2)
    side_to_modulate = int(effective_bias_norm > 0)
    modulation_amount = np.abs(effective_bias_norm) * 0.8
    control_signal[side_to_modulate] -= modulation_amount

    return control_signal
