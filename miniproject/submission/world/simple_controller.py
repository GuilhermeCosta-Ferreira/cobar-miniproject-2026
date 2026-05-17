# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from miniproject.simulation import MiniprojectSimulation
from flygym.examples.locomotion import TurningController


# ================================================================
# 1. Section: Functions
# ================================================================
class SimpleController:
    def __init__(self, sim: MiniprojectSimulation):
        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation, drives: np.ndarray | list) -> tuple:
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
