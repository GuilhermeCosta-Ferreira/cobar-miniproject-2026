# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from miniproject.simulation import MiniprojectSimulation



# ================================================================
# 1. Section: Functions
# ================================================================
def produce_human_view(sim: MiniprojectSimulation) -> np.ndarray:
    fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)

    return fly_vision
