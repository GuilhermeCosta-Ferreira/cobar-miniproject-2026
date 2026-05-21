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
from .olfaction import Olfaction
from .vision import (
    Vision,
    DragonflyAttackDetector
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "turning_inverse_model.joblib"



# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(
        self,
        sim: MiniprojectSimulation,
    ):
        self.turning_controller = HybridTurningController(sim.timestep)
        self.olfaction = Olfaction()
        self.wind = Wind(sim.mj_model)
        self.vision = Vision()

        self.dragonfly_detector = DragonflyAttackDetector(
            attack_threshold=0.06,
            hold_steps=10_000,
            min_consecutive_hits=1,
        )
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
        vision_velocity = np.array([0.0, 0.0])
        if ((current_step > 5e3) or self.vision.is_active) and sim.enable_grass:
            vision_velocity = self.vision.obstacle_to_velocity(sim, odor_velocity[0])

        velocity = odor_velocity + vision_velocity
        self._velocity_history.append(velocity)

        drives = self.inverse_model.predict(np.array([velocity]))[0]
        self._drive_history.append(drives)

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
