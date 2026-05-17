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
    alpha: float = 0.05
    current_signal: np.ndarray = field(
        default_factory=lambda: np.zeros(
            2,
        )
    )

    def process_olfaction(self, signal: np.ndarray) -> np.ndarray:
        """Applies a smoothing, when possible to olfactation"""
        if self.olfaction_smooth is None:
            self.olfaction_smooth = signal
        else:
            self.olfaction_smooth = (
                1 - self.alpha
            ) * self.olfaction_smooth + self.alpha * signal
        return self.olfaction_smooth
