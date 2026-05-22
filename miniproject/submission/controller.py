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
from .threat import DEFAULT_DRAGONFLY_STATE, DragonflyAttackDetector, EscapeController
from .config import load_config

MODEL_PATH = Path(__file__).resolve().parent / "periphery" / "models" / "turning_inverse_model_flat.joblib"
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "vision_config.yaml"
CONFIG = load_config(CONFIG_PATH)



# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(
        self,
        sim: MiniprojectSimulation,
        config: dict = CONFIG,
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
        self.dragonfly_detector = DragonflyAttackDetector.from_timestep(sim.timestep)
        self.escape_controller = EscapeController()
        self.current_escape_decision = self.escape_controller.last_decision

        self.current_drive = [0.0, 0.0]
        self.current_velocity = np.array([0.0, 0.0])
        self.wind_velocity = np.array([0.0, 0.0])
        self.obstacle_velocity = np.array([0.0, 0.0])
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
        wind_velocity = np.array([0.0, 0.0])
        if sim.enable_wind:
            wind = sim.get_antenna_data(sim.fly.name)
            wind_signal = self.wind.process_wind(
                wind,
                bias=0,
                lat_k=0.7,
                fwd_k=0.5,
            )
            wind_velocity = wind_signal_to_velocity(wind_signal)
        else:
            self.wind.current_signal = np.array([0.0, 0.0])
        self.wind_velocity = wind_velocity

        # VISION - dragonfly. This is perception-driven, not level-flag-driven.
        dragonfly_state = DEFAULT_DRAGONFLY_STATE.copy()
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
        self.current_escape_decision = escape_decision
        self.dragonfly_mode = escape_decision.mode
        self.dragonfly_danger_score = escape_decision.danger_score
        self.dragonfly_escape_direction = escape_decision.direction
        escape_config = self.escape_controller.config

        # VISION - obstacles. During dragonfly danger, obstacle vision is forced
        # on so a burst cannot ignore grass just because the normal gate is idle.
        vision_velocity = self.vision.obstacle_to_velocity(sim, odor_velocity[0])
        self.obstacle_velocity = vision_velocity

        if escape_decision.mode == "watch":
            # Walk cautiously while visible danger is far away, but keep the turn
            # sign aligned with normal odor tracking.
            watch_velocity = escape_decision.velocity.copy()
            watch_velocity[1] = escape_config.watch_odor_turn_gain * odor_velocity[1]
            velocity = (
                watch_velocity
                + escape_config.watch_obstacle_gain * vision_velocity
                + escape_config.watch_wind_gain * wind_velocity
            )
            velocity = adapt_velocity(
                velocity,
                max_forward=escape_config.watch_max_forward_velocity,
                max_turn=escape_config.watch_max_turn_velocity,
            )
        elif escape_decision.mode == "panic_escape":
            # Odor pursuit is suppressed during danger, but obstacle avoidance and
            # wind compensation still shape the escape vector.
            burst_velocity = escape_decision.velocity.copy()
            burst_velocity[1] = escape_config.panic_odor_turn_gain * odor_velocity[1]
            velocity = (
                burst_velocity
                + escape_config.burst_obstacle_gain * vision_velocity
                + escape_config.burst_wind_gain * wind_velocity
            )
            velocity = adapt_velocity(
                velocity,
                max_forward=escape_config.panic_max_forward_velocity,
                max_turn=escape_config.panic_max_turn_velocity,
            )
        else:
            velocity = odor_velocity + vision_velocity + wind_velocity
            velocity = drifter(
                current_velocity = velocity,
                dropoff_vt = self.dropoff_vt,
                max_vt = self.max_vt,
                max_vf = self.max_vf
            )

        self.current_velocity = velocity
        self._velocity_history.append(velocity)

        drives = self.inverse_model.predict(np.array([velocity]))[0]
        self.current_drive = drives
        self._drive_history.append(drives)

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
def adapt_velocity(
    velocity: np.ndarray,
    max_forward: float = 15.0,
    max_turn: float = 6.0,
) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=float)
    return np.array(
        [
            np.clip(velocity[0], -0.5 * max_forward, max_forward),
            np.clip(velocity[1], -max_turn, max_turn),
        ],
        dtype=float,
    )


def wind_signal_to_velocity(
    wind_signal: np.ndarray,
    forward_gain: float = 2.0,
    turn_gain: float = 1.2,
) -> np.ndarray:
    wind_signal = np.asarray(wind_signal, dtype=float)
    forward_velocity = forward_gain * np.mean(wind_signal)
    rotational_velocity = turn_gain * (wind_signal[1] - wind_signal[0])

    return np.array([forward_velocity, rotational_velocity], dtype=float)
