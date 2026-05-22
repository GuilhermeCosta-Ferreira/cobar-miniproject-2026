import numpy as np

from dataclasses import dataclass

from ..vision.detection import dragonfly_red_features_from_raw_vision

DEFAULT_DRAGONFLY_STATE: dict[str, float | bool] = {
    "visible": False,
    "attack": False,
    "red_score": 0.0,
    "largest_blob_frac": 0.0,
    "looming": 0.0,
    "side_bias": 0.0,
}


@dataclass(frozen=True)
class DragonflyDetectorConfig:
    visible_threshold: float = 0.0003
    visible_blob_threshold: float = 0.0005
    attack_threshold: float = 0.008
    blob_threshold: float = 0.002
    looming_threshold: float = 0.00045
    watch_hold_seconds: float = 1.5
    attack_hold_seconds: float = 0.3
    min_consecutive_hits: int = 1


class DragonflyAttackDetector:
    """
    Stateful detector.

    """

    def __init__(
        self,
        visible_threshold: float = DragonflyDetectorConfig.visible_threshold,
        visible_blob_threshold: float = DragonflyDetectorConfig.visible_blob_threshold,
        attack_threshold: float = DragonflyDetectorConfig.attack_threshold,
        blob_threshold: float = DragonflyDetectorConfig.blob_threshold,
        looming_threshold: float = DragonflyDetectorConfig.looming_threshold,
        watch_hold_steps: int = 2000,
        hold_steps: int = 0,
        min_consecutive_hits: int = DragonflyDetectorConfig.min_consecutive_hits,
    ):
        self.visible_threshold = visible_threshold
        self.visible_blob_threshold = visible_blob_threshold
        self.attack_threshold = attack_threshold
        self.blob_threshold = blob_threshold
        self.looming_threshold = looming_threshold
        self.watch_hold_steps = watch_hold_steps
        self.hold_steps = hold_steps
        self.min_consecutive_hits = min_consecutive_hits

        self.attack_memory_until = -1
        self.visible_memory_until = -1
        self.consecutive_hits = 0
        self.current_score = 0.0
        self.current_blob = 0.0
        self.current_looming = 0.0
        self.current_visible = False
        self.current_attack = False
        self.current_side_bias = 0.0
        self.attack_has_triggered = False

    @classmethod
    def from_timestep(
        cls,
        timestep: float,
        config: DragonflyDetectorConfig | None = None,
    ) -> "DragonflyAttackDetector":
        cfg = config or DragonflyDetectorConfig()
        return cls(
            visible_threshold=cfg.visible_threshold,
            visible_blob_threshold=cfg.visible_blob_threshold,
            attack_threshold=cfg.attack_threshold,
            blob_threshold=cfg.blob_threshold,
            looming_threshold=cfg.looming_threshold,
            watch_hold_steps=int(cfg.watch_hold_seconds / timestep),
            hold_steps=int(cfg.attack_hold_seconds / timestep),
            min_consecutive_hits=cfg.min_consecutive_hits,
        )

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

        visible_cue = (
            red_score > self.visible_threshold
            or largest_blob_frac > self.visible_blob_threshold
        )
        attack_cue = (
            red_score > self.attack_threshold
            and largest_blob_frac > self.blob_threshold
        ) or self.current_looming > self.looming_threshold

        if visible_cue:
            self.visible_memory_until = current_step + self.watch_hold_steps
            if side_bias != 0:
                self.current_side_bias = side_bias

        if attack_cue:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        threshold_reached = self.consecutive_hits >= self.min_consecutive_hits

        if threshold_reached and self.hold_steps <= 0:
            self.current_attack = True
            self.attack_has_triggered = True
            self.current_visible = (
                visible_cue or current_step < self.visible_memory_until
            )
            return self.current_attack

        if threshold_reached:
            self.attack_memory_until = current_step + self.hold_steps

        self.current_attack = current_step < self.attack_memory_until
        if self.current_attack:
            self.attack_has_triggered = True

        if self.attack_has_triggered and not self.current_attack and not visible_cue:
            self.visible_memory_until = -1

        self.current_visible = visible_cue or current_step < self.visible_memory_until

        return self.current_attack

    def detect_state_from_raw_vision(
        self,
        raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
        current_step: int,
        r_min: float = 90,
        dominance: float = 1.25,
        red_minus_green_min: float = 60,
    ) -> dict[str, float | bool]:
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

        return {
            "visible": self.current_visible,
            "attack": attack_active,
            "red_score": features["red_score"],
            "largest_blob_frac": features["largest_blob_frac"],
            "looming": self.current_looming,
            "side_bias": self.current_side_bias,
        }

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
        state = self.detect_state_from_raw_vision(
            raw_vision,
            current_step=current_step,
            r_min=r_min,
            dominance=dominance,
            red_minus_green_min=red_minus_green_min,
        )

        return state["attack"], state["red_score"], state["side_bias"]
