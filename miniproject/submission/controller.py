import numpy as np
from miniproject.simulation import MiniprojectSimulation
from flygym.examples.locomotion import TurningController

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)

        
        # get other observations as needed
        drives = np.array([1.0, 1.0])  # replace with your control logic
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
