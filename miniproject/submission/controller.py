# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path
from joblib import load

from miniproject.simulation import MiniprojectSimulation
from .hybrid_controller import HybridTurningController

from .wind import (
    Wind,
)
from .olfaction import (
    average_olfaction_signal,
    odor_intensity_to_control_signal,
    Olfaction,
)
from .vision import (
    obstacle_by_hue,
    produce_human_view,
    Vision,
)
from .threat import DragonflyAttackDetector, EscapeController

MODEL_PATH = Path(__file__).resolve().parent / "models" / "turning_inverse_model.joblib"


# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(
        self,
        sim: MiniprojectSimulation,
        vision_gain: float = 1
    ):
        self.turning_controller = HybridTurningController(sim.timestep)
        self.olfaction = Olfaction()
        self.wind = Wind(sim.mj_model)
        self.vision = Vision()

        self.frames = []
        self.vision_gain = vision_gain
        self.vision_signal = np.array([0.0, 0.0])
        self.dragonfly_visible = False
        self.dragonfly_attack = False
        self.dragonfly_red_score = 0.0
        self.dragonfly_blob = 0.0
        self.dragonfly_looming = 0.0
        self.dragonfly_side_bias = 0.0
        self.dragonfly_mode = "normal"
        self.dragonfly_danger_score = 0.0
        self.dragonfly_escape_direction = 0.0
        self.stability_score = 1.0
        self.is_unstable = False
        self.dragonfly_detector = DragonflyAttackDetector(
            visible_threshold=0.0003,
            visible_blob_threshold=0.0005,
            attack_threshold=0.004,
            blob_threshold=0.001,
            looming_threshold=0.00025,
            watch_hold_steps=int(1.5 / sim.timestep),
            hold_steps=int(0.35 / sim.timestep),
            min_consecutive_hits=1,
        )
        self.escape_controller = EscapeController()
        self.current_drive = [0.0, 0.0]
        self.inverse_model = load(MODEL_PATH)


    def step(self, sim: MiniprojectSimulation):
        current_step = sim._curr_step

        # OLFACTION
        smell = sim.get_olfaction(sim.fly.name)
        odor_velocity = self.olfaction.smell_to_velocity(smell)

        # WIND
        """
        lateral_olfaction = average_olfaction_signal(smooth_olfaction)
        odor_drives = odor_intensity_to_control_signal(lateral_olfaction, attractive_gain=-800)
        if sim.enable_wind:
            wind = sim.get_antenna_data(sim.fly.name)
            wind_signal = self.wind.process_wind(wind, bias=0, lat_k=2, fwd_k=2) # gain values heuristically set
            wind_signal = adapt_drives(wind_signal, max_signal = 0.5)
        else:
            wind_signal = np.array([0.0, 0.0])
        """

        # VISION
        vision_signal = np.array([0.0, 0.0])
        if ((current_step > 5e3) or self.vision.is_active) and sim.enable_grass:
            frame = produce_human_view(sim)
            vision_signal = obstacle_by_hue(frame, turn_gain=self.vision_gain)

            self.vision.add_signal(vision_signal)

        """
        # UPDATE THIS
        control_signals = odor_drives + vision_signal + wind_signal
        control_signals = adapt_drives(control_signals)
        """
        self.vision_signal = vision_signal

        # VISION - dragonfly. This is perception-driven, not level-flag-driven.
        dragonfly_state = self.dragonfly_detector.detect_state_from_raw_vision(
            raw_vision=sim.get_raw_vision(sim.fly.name),
            current_step=current_step,
        )
        self.dragonfly_visible = bool(dragonfly_state["visible"])
        self.dragonfly_attack = bool(dragonfly_state["attack"])
        self.dragonfly_red_score = float(dragonfly_state["red_score"])
        self.dragonfly_blob = float(dragonfly_state["largest_blob_frac"])
        self.dragonfly_looming = float(dragonfly_state["looming"])
        self.dragonfly_side_bias = float(dragonfly_state["side_bias"])
        self.vision.update_dragonfly_state(
            score=self.dragonfly_red_score,
            attack=self.dragonfly_attack,
        )

        escape_decision = self.escape_controller.step(sim, dragonfly_state)
        self.dragonfly_mode = escape_decision.mode
        self.dragonfly_danger_score = escape_decision.danger_score
        self.dragonfly_escape_direction = escape_decision.direction
        self.stability_score = escape_decision.stability_score
        self.is_unstable = escape_decision.unstable

        if escape_decision.mode == "recovery":
            control_signals = (
                escape_decision.drives
                + vision_signal
                + 0.5 * wind_signal
            )
            control_signals = adapt_drives(control_signals, max_signal=1.0)
        elif escape_decision.mode != "normal":
            # During dragonfly danger, odor pursuit is suppressed so the fly does
            # not turn away from a visible threat just to follow the banana plume.
            control_signals = (
                escape_decision.drives
                + vision_signal
                + 0.5 * wind_signal
            )
            max_signal = 2.3 if escape_decision.mode == "panic_escape" else 1.6
            control_signals = keep_escape_drives_forward(
                control_signals,
                escape_decision.mode,
            )
            control_signals = adapt_drives(control_signals, max_signal=max_signal)
        else:
            control_signals = odor_drives + vision_signal + wind_signal
            control_signals = adapt_drives(control_signals)

        drives = self.inverse_model.predict(np.array([odor_velocity]))[0]
        self.current_drive = drives

        joint_angles, adhesion = self.turning_controller.step(sim, drives)
        return joint_angles, adhesion


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def adapt_drives(drives: np.ndarray, max_signal: float = 2) -> np.ndarray:
    drives = np.asarray(drives, dtype=float)

    max_abs = np.max(np.abs(drives))

    if max_abs <= max_signal:
        return drives

    return drives / max_abs * max_signal


def keep_escape_drives_forward(drives: np.ndarray, mode: str) -> np.ndarray:
    drives = np.asarray(drives, dtype=float)

    min_drive_by_mode = {
        "watch": 0.15,
        "planned_escape": 0.55,
        "panic_escape": 1.05,
    }
    min_drive = min_drive_by_mode.get(mode, 0.0)

    return np.maximum(drives, min_drive)
