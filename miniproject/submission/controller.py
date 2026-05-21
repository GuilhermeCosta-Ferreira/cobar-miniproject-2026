# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

import matplotlib.pyplot as plt
from miniproject.simulation import MiniprojectSimulation
from flygym.examples.locomotion.turning_controller import TurningController
from .hybrid_controller import HybridTurningController

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

        # WIND (will update odor information)
        wind = sim.get_antenna_data(sim.fly.name)
        wind_signal = self.wind.process_wind(wind)

        #wind_x = get_wind_velocity(wind)

        # VISION
        if sim.enable_grass:
            frame = produce_human_view(sim)
            self.vision_signal = obstacle_by_hue(frame)
            odor_turn = odor_drives[0] - odor_drives[1]
            vision_turn = self.vision_signal[0] - self.vision_signal[1]
            if odor_turn != 0 and vision_turn != 0:
                if np.sign(odor_turn) != np.sign(vision_turn):
                    self.vision_signal *= -1
            self.vision.add_signal(self.vision_signal)
        else:
            self.vision_signal = np.array([0.0, 0.0])

        control_signals = odor_drives + self.vision_signal + wind_signal

        if sim.enable_terrain:
            drives = damp_drives_for_rough_terrain(control_signals)
        else:
            drives = np.clip(control_signals, 0.2, 1.3)

        joint_angles, adhesion = self.turning_controller.step(sim, drives)

        # DRAGONDFLY
        raw_vision = sim.get_raw_vision(sim.fly.name)

        dragonfly_attack, red_score = self.dragonfly_detector.detect_from_raw_vision(
            raw_vision=raw_vision,
            current_step=current_step,
        )

        self.vision.update_dragonfly_state(
            score=red_score,
            attack=dragonfly_attack,
        )

        if dragonfly_attack:
            adhesion = np.ones_like(adhesion)
        
        return joint_angles, adhesion
