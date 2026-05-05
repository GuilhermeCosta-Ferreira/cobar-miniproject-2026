# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import cv2

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

def produce_raw_vision(sim: MiniprojectSimulation) -> np.ndarray:
    """Produce the raw camera view at the current step to be ploted"""
    vision = sim.get_raw_vision(sim.fly.name)
    return vision

def _overlay_mask(im: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]=(100, 100, 0)) -> np.ndarray:
    """Overlay mask on the image"""
    mask_color = np.zeros_like(im)
    mask_color[mask] = color
    overlay = cv2.addWeighted(im, 1, mask_color, 0.5, 0)
    return overlay

    