# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

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
    DragonflyAttackDetector
)

# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        self.turning_controller = HybridTurningController(sim.timestep)
        self.olfaction = Olfaction()
        self.wind = Wind(sim.mj_model)
        self.vision = Vision()
        self.frames = []
        self.vision_signal = np.array([0.0, 0.0])
        self.dragonfly_detector = DragonflyAttackDetector(
            attack_threshold=0.06,
            hold_steps=10_000,
            min_consecutive_hits=1,
        )


    def step(self, sim: MiniprojectSimulation):
        current_step = sim._curr_step

        # OLFACTION
        olfaction = sim.get_olfaction(sim.fly.name)
        smooth_olfaction = self.olfaction.process_olfaction(olfaction)
        lateral_olfaction = average_olfaction_signal(smooth_olfaction)
        odor_drives = odor_intensity_to_control_signal(lateral_olfaction, attractive_gain=-800)
        self.olfaction.current_signal = odor_drives

        
        # WIND
        wind = sim.get_antenna_data(sim.fly.name)
        wind_signal = self.wind.process_wind(wind, bias=0, lat_k=2, fwd_k=2) # gain values heuristically set
        

        # VISION
        if sim.enable_grass:
            frame = produce_human_view(sim)
            vision_signal = obstacle_by_hue(frame)

            self.vision.add_signal(vision_signal)

        # UPDATE THIS
        #updated_olfaction = update_olfaction(lateral_olfactation, wind_x)
        #control_signals = odor_drives + vision_signal + wind_signal
        control_signals = odor_drives + vision_signal
        control_signals = adapt_drives(control_signals)
        #control_signals = odor_drives

        #drives = damp_drives_for_rough_terrain(control_signals)
        joint_angles, adhesion = self.turning_controller.step(sim, control_signals)
        #joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion


def adapt_drives(drives: np.ndarray, max_signal: float = 2) -> np.ndarray:
    drives = np.asarray(drives, dtype=float)

    max_abs = np.max(np.abs(drives))

    if max_abs <= max_signal:
        return drives

    return drives / max_abs * max_signal
