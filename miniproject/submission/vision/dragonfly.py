import numpy as np

from .detection import dragonfly_red_score_from_raw_vision


class DragonflyAttackDetector:
    """
    Stateful detector.

    """

    def __init__(
        self,
        attack_threshold: float = 0.06,
        hold_steps: int = 10_000,
        min_consecutive_hits: int = 1,
    ):
        self.attack_threshold = attack_threshold
        self.hold_steps = hold_steps
        self.min_consecutive_hits = min_consecutive_hits

        self.attack_memory_until = -1
        self.consecutive_hits = 0
        self.current_score = 0.0
        self.current_attack = False

    def update(self, red_score: float, current_step: int) -> bool:
        """
        Update attack state from current red score.
        """
        self.current_score = red_score

        if red_score > self.attack_threshold:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        if self.consecutive_hits >= self.min_consecutive_hits:
            self.attack_memory_until = current_step + self.hold_steps

        self.current_attack = current_step < self.attack_memory_until

        return self.current_attack

    def detect_from_raw_vision(
        self,
        raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
        current_step: int,
        r_min: float = 90,
        dominance: float = 1.25,
        red_minus_green_min: float = 20,
    ) -> tuple[bool, float]:
        """
        Compute red score from raw vision, then update attack memory.
        """
        red_score, _ = dragonfly_red_score_from_raw_vision(
            raw_vision,
            r_min=r_min,
            dominance=dominance,
            red_minus_green_min=red_minus_green_min,
        )

        attack_active = self.update(red_score, current_step)

        return attack_active, red_score