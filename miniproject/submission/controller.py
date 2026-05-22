# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path
from joblib import load

from miniproject.simulation import MiniprojectSimulation
#from flygym.examples.locomotion import TurningController
from .hybrid_controller import HybridTurningController

from .wind import (
    Wind,
)
from .olfaction import Olfaction
from .vision import (
    Vision,
)
from .threat import DEFAULT_DRAGONFLY_STATE, DragonflyAttackDetector, EscapeController

MODEL_PATH = Path(__file__).resolve().parent / "periphery" / "models" / "turning_inverse_model_flat.joblib"


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
        #self.turning_controller = TurningController(sim.timestep)
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

        # OLFACTION
        smell = sim.get_olfaction(sim.fly.name)
        odor_velocity = self.olfaction.smell_to_velocity(smell)

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
        vision_velocity = np.array([0.0, 0.0])
        should_check_obstacles = (
            ((current_step > 5e3) or self.vision.is_active)
            or escape_decision.mode in ("watch", "panic_escape")
        )
        if should_check_obstacles and sim.enable_grass:
            vision_velocity = self.vision.obstacle_to_velocity(sim, odor_velocity[0])
        else:
            self.vision.current_signal = vision_velocity
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
            if vision_velocity[0] < 0.0:
                velocity[0] += escape_config.burst_obstacle_brake_gain * vision_velocity[0]
            velocity = adapt_velocity(
                velocity,
                max_forward=escape_config.panic_max_forward_velocity,
                max_turn=escape_config.panic_max_turn_velocity,
            )
        else:
            velocity = odor_velocity + vision_velocity + wind_velocity
            velocity = adapt_velocity(velocity, max_forward=15.0, max_turn=9.0)

        self.current_velocity = velocity
        self._velocity_history.append(velocity)

        drives = self.inverse_model.predict(np.array([velocity]))[0]
        self.current_drive = drives
        self._drive_history.append(drives)

        joint_angles, adhesion = self.turning_controller.step(sim, drives)
        #joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
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
