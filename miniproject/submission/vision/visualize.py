# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from miniproject.simulation import MiniprojectSimulation



# ================================================================
# 1. Section: Functions
# ================================================================
def produce_fly_view(sim: MiniprojectSimulation) -> np.ndarray:
    """Produce the fly view at the current step to be ploted"""
    ommatidia_readouts = sim.get_ommatidia_readouts(sim.fly.name)
    im = np.concatenate(
        [
            sim.fly.retina.hex_pxls_to_human_readable(eye.max(-1), color_8bit=True)
            for eye in ommatidia_readouts
        ],
        axis=1,
    )

    return im
