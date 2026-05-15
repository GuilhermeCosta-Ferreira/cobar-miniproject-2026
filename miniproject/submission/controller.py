# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from miniproject.simulation import MiniprojectSimulation
from flygym.examples.locomotion.turning_controller import TurningController

from .turning_controller import damp_drives_for_rough_terrain
from .wind import (
    Wind,
    get_wind_velocity,
    update_olfaction
)
from .olfactation import (
    average_olfaction_signal,
    odor_intensity_to_control_signal,
    Olfaction
)
from .vision import obstacle_by_hue, produce_human_view



# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        self.turning_controller = TurningController(sim.timestep)
        self.olfaction = Olfaction()
        self.wind = Wind()
        self.vision = Vision()

    def step(self, sim: MiniprojectSimulation):
        current_step = sim._curr_step

        # OLFACTION
        olfaction = sim.get_olfaction(sim.fly.name)
        smooth_olfaction = self.olfaction.process_olfaction(olfaction)
        lateral_olfactation = average_olfaction_signal(smooth_olfaction)
        odor_drives = odor_intensity_to_control_signal(lateral_olfactation, attractive_gain=-800)
        self.olfaction.current_signal = odor_drives

        # WIND (will update odor information)
        #wind = sim.get_antenna_data(sim.fly.name)
        #wind_x = get_wind_velocity(wind)

        # VISION
        vision_signal = np.array([0.0, 0.0])
        if ((current_step > 5e3) and (current_step % 1e3)) or self.vision.is_active:
            frame = produce_human_view(sim)
            vision_signal = obstacle_by_hue(frame)

            self.vision.add_signal(vision_signal)

        # UPDATE THIS
        #updated_olfaction = update_olfaction(lateral_olfactation, wind_x)
        control_signals = odor_drives + vision_signal

        drives = damp_drives_for_rough_terrain(control_signals)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion


class Vision():
    def __init__(self, max_size: int = 100):
        self.signal_history: list[np.ndarray] = []
        self.max_size = max_size
        self.current_signal = [0.0, 0.0]

    @property
    def history_size(self) -> int:
        return len(self.signal_history)

    @property
    def history_sum(self) -> float:
        return np.sum(self.signal_history)

    def add_signal(self, signal: np.ndarray) -> None:
        self.signal_history.append(signal)
        self.current_signal = signal

        if self.history_size > self.max_size:
            self.signal_history.pop(0)

    @property
    def is_active(self) -> bool:
        if self.history_size == self.max_size:
            return self.history_sum > 0

        return False
