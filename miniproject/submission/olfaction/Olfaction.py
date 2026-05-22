"""Module resposible for olfaction control, this is the main driver of behaviour"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass, field
from .signal_processing import get_smooth_olfaction, get_average_signal
from .velocity import get_velocity_vector


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Olfaction:
    olfaction_smooth: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    alpha: float = 0.05

    _intensity_hist: list = field(default_factory=list)
    _velocity_history: list = field(default_factory=list)

    # ================================================================
    # 2. Section: Properties
    # ================================================================
    @property
    def intensity_hist(self):
        return np.asarray(self._intensity_hist)

    @property
    def velocity_hist(self):
        return np.asarray(self._velocity_history)

    # ================================================================
    # 3. Section: Methods
    # ================================================================
    def smell_to_velocity(
        self, signal: np.ndarray, current_vf: float, max_vt: float, gain: float
    ) -> np.ndarray:
        # 1. Temporal smooths it to avoid big oscilations
        self.olfaction_smooth = get_smooth_olfaction(
            self.olfaction_smooth, signal, self.alpha
        )

        # 2. Averages it over the lateralizaed antena
        lat_olfaction = get_average_signal(self.olfaction_smooth)
        self._intensity_hist.append(lat_olfaction)

        # 3. Converts into velocities
        odor_velocity = get_velocity_vector(
            lat_olfaction=lat_olfaction,
            forward_velocity=current_vf,
            max_turn_velocity=max_vt,
            gain=gain,
        )
        self._velocity_history.append(odor_velocity)

        return odor_velocity
