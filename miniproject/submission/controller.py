# ================================================================
# 0. Section: IMPORTS
# ================================================================
from miniproject.simulation import MiniprojectSimulation

# our imports
from .odor_tracking import odor_intensity_to_control_signal
from flygym.examples.locomotion.turning_controller import TurningController
from .rough_terrain import damp_drives_for_rough_terrain



# ================================================================
# 1. Section: Controler Class
# ================================================================
class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        self.turning_controller = TurningController(sim.timestep)

        # our inits
        self.olfaction_smooth = None
        self.alpha = 0.05


    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        # smooth signal
        self.process_olfaction(olfaction)

        # get control signals from olfaction
        control_signals = self.olfaction_smooth

        odor_drives = odor_intensity_to_control_signal(control_signals, attractive_gain=-800)
        drives = damp_drives_for_rough_terrain(odor_drives)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion

    def process_olfaction(self, signal):
        if self.olfaction_smooth is None:
            self.olfaction_smooth = signal
        else:
            self.olfaction_smooth = (
                (1 - self.alpha) * self.olfaction_smooth + self.alpha * signal
            )
        return self.olfaction_smooth
