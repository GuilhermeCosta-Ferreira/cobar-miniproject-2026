from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DangerConfig:
    area_scale: float = 0.006
    red_scale: float = 0.012
    expansion_scale: float = 0.0008
    visible_weight: float = 0.10
    area_weight: float = 0.40
    expansion_weight: float = 0.35
    red_weight: float = 0.15
    panic_threshold: float = 0.55


@dataclass
class EscapeConfig:
    danger: DangerConfig = field(default_factory=DangerConfig)
    watch_forward_velocity: float = 3.5
    watch_turn_velocity: float = 1.5
    watch_odor_turn_gain: float = 0.12
    watch_obstacle_gain: float = 0.9
    watch_wind_gain: float = 0.4
    watch_max_forward_velocity: float = 5.0
    watch_max_turn_velocity: float = 1.0
    panic_forward_velocity: float = 18.0
    panic_turn_velocity: float = 2.0
    panic_odor_turn_gain: float = 0.15
    burst_obstacle_gain: float = 2.4
    burst_wind_gain: float = 0.5
    panic_max_forward_velocity: float = 20.0
    panic_max_turn_velocity: float = 2.5
    watch_drive: float = 0.25
    panic_forward_drive: float = 1.9
    panic_turn_drive: float = 0.8
    centered_side_threshold: float = 0.15


@dataclass
class EscapeDecision:
    mode: str
    danger_score: float
    direction: float
    velocity: np.ndarray
    drives: np.ndarray


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def compute_danger_score(
    dragonfly_state: dict[str, float | bool],
    config: DangerConfig | None = None,
) -> float:
    """Combine red area, looming growth, and detector confidence into one score."""
    cfg = config or DangerConfig()
    visible = bool(dragonfly_state.get("visible", False))
    attack = bool(dragonfly_state.get("attack", False))
    red_score = float(dragonfly_state.get("red_score", 0.0))
    area = float(dragonfly_state.get("largest_blob_frac", 0.0))
    expansion = float(dragonfly_state.get("looming", 0.0))

    normalized_area = _clip01(area / cfg.area_scale)
    normalized_red = _clip01(red_score / cfg.red_scale)
    normalized_expansion = _clip01(max(0.0, expansion) / cfg.expansion_scale)
    confidence = 1.0 if visible else 0.0

    score = (
        cfg.area_weight * normalized_area
        + cfg.expansion_weight * normalized_expansion
        + cfg.red_weight * normalized_red
        + cfg.visible_weight * confidence
    )

    if attack:
        score = max(score, cfg.panic_threshold)

    return _clip01(score)


def compute_escape_direction(
    dragonfly_state: dict[str, float | bool],
    centered_side_threshold: float = 0.15,
) -> float:
    """
    Return -1 for a leftward escape, +1 for a rightward escape, and 0 for straight.

    """
    side_bias = float(dragonfly_state.get("side_bias", 0.0))

    if abs(side_bias) < centered_side_threshold:
        return 0.0

    return -float(np.sign(side_bias))


def select_escape_mode(
    dragonfly_state: dict[str, float | bool],
    danger_score: float,
    config: DangerConfig | None = None,
) -> str:
    cfg = config or DangerConfig()

    if not bool(dragonfly_state.get("visible", False)) and danger_score <= 0.0:
        return "normal"

    if (
        bool(dragonfly_state.get("attack", False))
        or danger_score >= cfg.panic_threshold
    ):
        return "panic_escape"

    return "watch"


def compute_escape_drives(
    mode: str,
    direction: float,
    config: EscapeConfig | None = None,
) -> np.ndarray:
    """
    Convert an escape mode and direction into left/right descending drives.

    Lower left and higher right drive produces a leftward turn in the same drive
    convention used by the odor controller.
    """
    cfg = config or EscapeConfig()

    if mode == "watch":
        return np.array([cfg.watch_drive, cfg.watch_drive], dtype=float)

    if mode == "panic_escape":
        base = cfg.panic_forward_drive
        turn = cfg.panic_turn_drive
    else:
        return np.zeros(2, dtype=float)

    if direction < 0:
        return np.array([max(0.0, base - turn), base + turn], dtype=float)
    if direction > 0:
        return np.array([base + turn, max(0.0, base - turn)], dtype=float)

    return np.array([base, base], dtype=float)


def compute_escape_velocity(
    mode: str,
    direction: float,
    config: EscapeConfig | None = None,
) -> np.ndarray:
    """
    Convert an escape mode and direction into [forward, rotational] velocity.

    """
    cfg = config or EscapeConfig()

    if mode == "watch":
        return np.array([cfg.watch_forward_velocity, 0.0], dtype=float)

    if mode == "panic_escape":
        forward = cfg.panic_forward_velocity
        turn = cfg.panic_turn_velocity
    else:
        return np.zeros(2, dtype=float)

    return np.array([forward, -direction * turn], dtype=float)


class EscapeController:
    def __init__(self, config: EscapeConfig | None = None):
        self.config = config or EscapeConfig()
        self.last_decision = EscapeDecision(
            mode="normal",
            danger_score=0.0,
            direction=0.0,
            velocity=np.zeros(2, dtype=float),
            drives=np.zeros(2, dtype=float),
        )

    def step(self, sim, dragonfly_state: dict[str, float | bool]) -> EscapeDecision:
        danger_score = compute_danger_score(dragonfly_state, self.config.danger)
        mode = select_escape_mode(dragonfly_state, danger_score, self.config.danger)
        direction = compute_escape_direction(
            dragonfly_state,
            centered_side_threshold=self.config.centered_side_threshold,
        )
        velocity = compute_escape_velocity(mode, direction, self.config)
        drives = compute_escape_drives(mode, direction, self.config)

        decision = EscapeDecision(
            mode=mode,
            danger_score=danger_score,
            direction=direction,
            velocity=velocity,
            drives=drives,
        )
        self.last_decision = decision
        return decision
