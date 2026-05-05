# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from miniproject.simulation import MiniprojectSimulation
from flygym.examples.locomotion.turning_controller import TurningController

from .Olfaction import Olfaction
from .odor_tracking import odor_intensity_to_control_signal
from .rough_terrain import damp_drives_for_rough_terrain



# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        self.turning_controller = TurningController(sim.timestep)
        self.olfaction = Olfaction()

    def step(self, sim: MiniprojectSimulation):
        # OLFACTION
        olfaction = sim.get_olfaction(sim.fly.name)
        smooth_olfaction = self.olfaction.process_olfaction(olfaction)

        # WIND
        wind = sim.get_antenna_data(sim.fly.name)

        control_signals = smooth_olfaction

        odor_drives = odor_intensity_to_control_signal(control_signals, attractive_gain=-800)
        drives = damp_drives_for_rough_terrain(odor_drives)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
