# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

import matplotlib.pyplot as plt
from miniproject.simulation import MiniprojectSimulation
from flygym.examples.locomotion.turning_controller import TurningController

from .turning_controller import damp_drives_for_rough_terrain
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
    visualize,
    Vision, 
)

# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.olfaction = Olfaction()
        self.wind = Wind(sim.mj_model)
        self.vision = Vision()        
        self.frames = []


    def step(self, sim: MiniprojectSimulation):
        current_step = sim._curr_step

        # OLFACTION
        olfaction = sim.get_olfaction(sim.fly.name)
        smooth_olfaction = self.olfaction.process_olfaction(olfaction)
        lateral_olfaction = average_olfaction_signal(smooth_olfaction)
        odor_drives = odor_intensity_to_control_signal(lateral_olfaction, attractive_gain=-800)
        self.olfaction.current_signal = odor_drives

        # WIND (will update odor information)
        wind = sim.get_antenna_data(sim.fly.name)
        wind_signal = self.wind.process_wind(wind)
        
        #wind_x = get_wind_velocity(wind)

        # VISION
        vision_signal = np.array([0.0, 0.0])
        if ((current_step > 5e3) and (current_step % 1e3)) or self.vision.is_active:
            frame = produce_human_view(sim)
            vision_signal = obstacle_by_hue(frame)

            self.vision.add_signal(vision_signal)

        # UPDATE THIS
        #updated_olfaction = update_olfaction(lateral_olfactation, wind_x)
        control_signals = odor_drives + vision_signal + wind_signal

        drives = damp_drives_for_rough_terrain(control_signals)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
