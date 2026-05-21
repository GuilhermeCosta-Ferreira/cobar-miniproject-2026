import numpy as np

from .detection import dragonfly_red_features_from_raw_vision


class DragonflyAttackDetector:
    """
    Stateful detector.

    """

    def __init__(
        self,
        attack_threshold: float = 0.04,
        blob_threshold: float = 0.02,
        looming_threshold: float = 0.004,
        hold_steps: int = 10_000,
        min_consecutive_hits: int = 1,
    ):
        self.attack_threshold = attack_threshold
        self.blob_threshold = blob_threshold
        self.looming_threshold = looming_threshold
        self.hold_steps = hold_steps
        self.min_consecutive_hits = min_consecutive_hits

        self.attack_memory_until = -1
        self.consecutive_hits = 0
        self.current_score = 0.0
        self.current_blob = 0.0
        self.current_looming = 0.0
        self.current_attack = False
        self.current_side_bias = 0.0

    def update(
        self,
        red_score: float,
        largest_blob_frac: float,
        current_step: int,
        side_bias: float = 0.0,
    ) -> bool:
        """
        Update attack state from current red score.
        """
        self.current_looming = max(0.0, largest_blob_frac - self.current_blob)
        self.current_score = red_score
        self.current_blob = largest_blob_frac

        attack_cue = (
            (red_score > self.attack_threshold and largest_blob_frac > self.blob_threshold)
            or self.current_looming > self.looming_threshold
        )

        if attack_cue:
            self.consecutive_hits += 1
            if side_bias != 0:
                self.current_side_bias = side_bias
        else:
            self.consecutive_hits = 0

        threshold_reached = self.consecutive_hits >= self.min_consecutive_hits

        if threshold_reached and self.hold_steps <= 0:
            self.current_attack = True
            return self.current_attack

        if threshold_reached:
            self.attack_memory_until = current_step + self.hold_steps

        self.current_attack = current_step < self.attack_memory_until

        return self.current_attack

    def detect_from_raw_vision(
        self,
        raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
        current_step: int,
        r_min: float = 90,
        dominance: float = 1.25,
        red_minus_green_min: float = 60,
    ) -> tuple[bool, float, float]:
        """
        Compute red score from raw vision, then update attack memory.
        """
        features = dragonfly_red_features_from_raw_vision(
            raw_vision,
            r_min=r_min,
            dominance=dominance,
            red_minus_green_min=red_minus_green_min,
        )

        attack_active = self.update(
            red_score=features["red_score"],
            largest_blob_frac=features["largest_blob_frac"],
            current_step=current_step,
            side_bias=features["side_bias"],
        )

        return attack_active, features["red_score"], self.current_side_bias
