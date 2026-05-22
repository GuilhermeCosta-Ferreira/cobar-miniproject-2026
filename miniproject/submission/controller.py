# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path
from joblib import load

from miniproject.simulation import MiniprojectSimulation
from .hybrid_controller import HybridTurningController

from .wind import Wind
from .olfaction import Olfaction
from .vision import Vision
from .threat import DragonflyAttackDetector, EscapeController

MODEL_PATH = Path(__file__).resolve().parent / "periphery" / "models" / "turning_inverse_model_flat.joblib"



# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(
        self,
        sim: MiniprojectSimulation,
        config: dict,
    ):
        self.config = config

        self.base_vf = config["controller"]["base_vf"]
        self.max_vt = config["controller"]["max_vt"]
        self.max_vf = config["controller"]["max_vf"]
        self.dropoff_vt = config["controller"]["dropoff_vt"]

        self.olfaction_gain = config["olfaction"]["gain"]

        hybrid_cfg = config["hybrid"]
        f_cfg = hybrid_cfg["f"]
        m_cfg = hybrid_cfg["m"]
        h_cfg = hybrid_cfg["h"]
        self.max_increment = config["hybrid_max_increment"]

        self.correction_vectors = {
            "f": np.array([
                f_cfg["coxa_lift"],
                f_cfg["coxa_roll"],
                f_cfg["coxa_yaw"],
                f_cfg["femur_lift"],
                f_cfg["femur_roll"],
                f_cfg["tibia_lift"],
                f_cfg["tarsus1_lift"],
            ]),
            "m": np.array([
                m_cfg["coxa_lift"],
                m_cfg["coxa_roll"],
                m_cfg["coxa_yaw"],
                m_cfg["femur_lift"],
                m_cfg["femur_roll"],
                m_cfg["tibia_lift"],
                m_cfg["tarsus1_lift"],
            ]),
            "h": np.array([
                h_cfg["coxa_lift"],
                h_cfg["coxa_roll"],
                h_cfg["coxa_yaw"],
                h_cfg["femur_lift"],
                h_cfg["femur_roll"],
                h_cfg["tibia_lift"],
                h_cfg["tarsus1_lift"],
            ]),
        }

        self.turning_controller = HybridTurningController(
            sim.timestep,
            max_increment = self.max_increment,
            correction_vectors = self.correction_vectors
        )
        self.olfaction = Olfaction()
        self.wind = Wind(sim.mj_model)
        self.vision = Vision(config["vision"])

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

        self._velocity_history: list = []
        self._drive_history: list = []




    # ================================================================
    # 2. Section: Properties
    # ================================================================
    @property
    def velocity_hist(self):
        return np.asarray(self._velocity_history)

    @property
    def drive_hist(self):
        return np.asarray(self._drive_history)



    # ================================================================
    # 3. Section: Methods
    # ================================================================
    def step(self, sim: MiniprojectSimulation):
        current_step = sim._curr_step
        current_vf = self.base_vf

        # OLFACTION
        smell = sim.get_olfaction(sim.fly.name)
        odor_velocity = self.olfaction.smell_to_velocity(
            smell, current_vf, self.max_vt, self.olfaction_gain
        )

        # WIND
        """
        if sim.enable_wind:
            wind = sim.get_antenna_data(sim.fly.name)
            wind_signal = self.wind.process_wind(wind, bias=0, lat_k=2, fwd_k=2) # gain values heuristically set
            wind_signal = adapt_drives(wind_signal, max_signal = 0.5)
        else:
            wind_signal = np.array([0.0, 0.0])
        """

        # VISION
        vision_velocity = np.array([0.0, 0.0])
        if sim.enable_grass:
            vision_velocity = self.vision.obstacle_to_velocity(
                sim = sim,
                current_forward_vel = odor_velocity[0],
            )

        velocity = odor_velocity + vision_velocity
        velocity = drifter(
            current_velocity = velocity,
            dropoff_vt = self.dropoff_vt,
            max_vt = self.max_vt,
            max_vf = self.max_vf
        )
        self._velocity_history.append(velocity)

        drives = self.inverse_model.predict(np.array([velocity]))[0]
        self._drive_history.append(drives)

        # VISION - dragonfly. This is perception-driven, not level-flag-driven.
        """
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
            control_signals = (escape_decision.drives + drives)
            control_signals = adapt_drives(control_signals, max_signal=1.0)
        elif escape_decision.mode != "normal":
            # During dragonfly danger, odor pursuit is suppressed so the fly does
            # not turn away from a visible threat just to follow the banana plume.
            control_signals = (escape_decision.drives + drives)
            max_signal = 2.3 if escape_decision.mode == "panic_escape" else 1.6
            control_signals = keep_escape_drives_forward(
                control_signals,
                escape_decision.mode,
            )
            control_signals = adapt_drives(control_signals, max_signal=max_signal)
        else:
            control_signals = drives
        """

        joint_angles, adhesion = self.turning_controller.step(sim, drives)
        return joint_angles, adhesion


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def drifter(
    current_velocity: np.ndarray,
    dropoff_vt: float,
    max_vt: float,
    max_vf: float
) -> np.ndarray:
    current_vf = current_velocity[0]
    current_vt = current_velocity[1]

    current_vf = np.min([current_vf, max_vf])

    vt_abs = abs(current_vt)

    if max_vt <= dropoff_vt:
        raise ValueError("max_vt must be larger than dropoff_vt.")

    if vt_abs <= dropoff_vt:
        vf_scale = 1.0
    elif vt_abs >= max_vt:
        vf_scale = 0.0
    else:
        vf_scale = 1.0 - (vt_abs - dropoff_vt) / (max_vt - dropoff_vt)

    drifted_vf = current_vf * vf_scale

    return np.array([drifted_vf, current_vt])


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
