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
    planned_threshold: float = 0.18
    panic_threshold: float = 0.60


@dataclass
class EscapeConfig:
    danger: DangerConfig = field(default_factory=DangerConfig)
    watch_drive: float = 0.25
    planned_forward_drive: float = 1.1
    planned_turn_drive: float = 0.55
    panic_forward_drive: float = 1.9
    panic_turn_drive: float = 0.8
    centered_side_threshold: float = 0.15
    unstable_up_threshold: float = 0.35
    stable_up_threshold: float = 0.55
    recovery_drive: float = 0.25


@dataclass
class EscapeDecision:
    mode: str
    danger_score: float
    direction: float
    drives: np.ndarray
    stability_score: float
    unstable: bool


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

    if danger_score >= cfg.planned_threshold:
        return "planned_escape"

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

    if mode == "planned_escape":
        base = cfg.planned_forward_drive
        turn = cfg.planned_turn_drive
    elif mode == "panic_escape":
        base = cfg.panic_forward_drive
        turn = cfg.panic_turn_drive
    elif mode == "recovery":
        return np.array([cfg.recovery_drive, cfg.recovery_drive], dtype=float)
    else:
        return np.zeros(2, dtype=float)

    if direction < 0:
        return np.array([max(0.0, base - turn), base + turn], dtype=float)
    if direction > 0:
        return np.array([base + turn, max(0.0, base - turn)], dtype=float)

    return np.array([base, base], dtype=float)


def quat_to_up_z(quat: np.ndarray | list[float]) -> float:
    """Return the world z-component of the body +z axis for a MuJoCo wxyz quat."""
    w, x, y, z = np.asarray(quat, dtype=float)
    return float(1.0 - 2.0 * (x * x + y * y))


def compute_body_stability(
    sim,
    body_reference: str = "c_thorax",
    unstable_up_threshold: float = 0.35,
    stable_up_threshold: float = 0.55,
) -> tuple[float, bool]:
    """
    Estimate whether the fly is upright enough to safely execute an escape burst.
    """
    body_segments = sim.fly.get_bodysegs_order()
    body_names = [
        seg.name if hasattr(seg, "name") else str(seg) for seg in body_segments
    ]

    if body_reference in body_names:
        body_idx = body_names.index(body_reference)
    else:
        root_name = (
            sim.fly.root_segment.name
            if hasattr(sim.fly.root_segment, "name")
            else str(sim.fly.root_segment)
        )
        body_idx = body_names.index(root_name) if root_name in body_names else 0

    quat = sim.get_body_rotations(sim.fly.name)[body_idx]
    up_z = quat_to_up_z(quat)
    stability_score = _clip01(
        (up_z - unstable_up_threshold) / (stable_up_threshold - unstable_up_threshold)
    )

    return stability_score, up_z < unstable_up_threshold


class EscapeController:
    def __init__(self, config: EscapeConfig | None = None):
        self.config = config or EscapeConfig()
        self.last_decision = EscapeDecision(
            mode="normal",
            danger_score=0.0,
            direction=0.0,
            drives=np.zeros(2, dtype=float),
            stability_score=1.0,
            unstable=False,
        )

    def step(self, sim, dragonfly_state: dict[str, float | bool]) -> EscapeDecision:
        stability_score, unstable = compute_body_stability(
            sim,
            unstable_up_threshold=self.config.unstable_up_threshold,
            stable_up_threshold=self.config.stable_up_threshold,
        )

        if unstable:
            decision = EscapeDecision(
                mode="recovery",
                danger_score=0.0,
                direction=0.0,
                drives=compute_escape_drives("recovery", 0.0, self.config),
                stability_score=stability_score,
                unstable=True,
            )
            self.last_decision = decision
            return decision

        danger_score = compute_danger_score(dragonfly_state, self.config.danger)
        mode = select_escape_mode(dragonfly_state, danger_score, self.config.danger)
        direction = compute_escape_direction(
            dragonfly_state,
            centered_side_threshold=self.config.centered_side_threshold,
        )
        drives = compute_escape_drives(mode, direction, self.config)

        decision = EscapeDecision(
            mode=mode,
            danger_score=danger_score,
            direction=direction,
            drives=drives,
            stability_score=stability_score,
            unstable=False,
        )
        self.last_decision = decision
        return decision
