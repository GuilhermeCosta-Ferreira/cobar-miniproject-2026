# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass, field


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Olfaction:
    olfaction_smooth: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    current_signal: np.ndarray = field(default_factory=lambda: np.zeros(2,))
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
    def process_olfaction(self, signal: np.ndarray) -> np.ndarray:
        """Applies a smoothing, when possible to olfactation"""
        if self.olfaction_smooth is None:
            self.olfaction_smooth = signal
        else:
            self.olfaction_smooth = (
                1 - self.alpha
            ) * self.olfaction_smooth + self.alpha * signal
        return self.olfaction_smooth

    def average_olfaction_signal(self) -> np.ndarray:
        """Assumes there is only one type of smell and averages over each antenna"""
        average = np.average(self.olfaction_smooth[:, 0].reshape(2, 2), axis=0, weights=[9, 1])
        self._intensity_hist.append(average)
        return average

    def build_drive_vector(
        self,
        lat_olfaction: np.ndarray,
        forward_velocity: float,
        max_turn_velocity: float,
        min_forward_velocity: float
    ) -> np.ndarray:
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
        self._velocity_history.append(velocity)
        return velocity
