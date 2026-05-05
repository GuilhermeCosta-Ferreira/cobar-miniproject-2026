import numpy as np
import matplotlib.pyplot as plt
from miniproject.simulation import MiniprojectSimulation

# our imports
from .odor_tracking import odor_intensity_to_control_signal
from .vision import visualize

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        from matplotlib.animation import FuncAnimation
        # you may also implement your own turning controller
        self.turning_controller = TurningController(sim.timestep)

        # our inits
        self.olfaction_smooth = None
        self.alpha = 0.001
        self.frames = []


    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        # smooth signal
        self.process_olfaction(olfaction)

        # get control signals from olfaction
        control_signals = self.olfaction_smooth

        drives = odor_intensity_to_control_signal(control_signals)
        joint_angles, adhesion = self.turning_controller.step(drives)
        self.frames.append(visualize.produce_fly_view(sim))
        return joint_angles, adhesion

    def process_olfaction(self, signal):
        if self.olfaction_smooth is None:
            self.olfaction_smooth = signal
        else:
            self.olfaction_smooth = (
                (1 - self.alpha) * self.olfaction_smooth + self.alpha * signal
            )
        return self.olfaction_smooth
    