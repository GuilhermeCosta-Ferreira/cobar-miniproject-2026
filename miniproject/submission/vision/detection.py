# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path

from .unpack import prepare_image_for_png
from .obstacles import get_obstacles_by_height, get_signals_from_centroids
from .hsv import convert_to_hsv, hue_to_degree, get_hsv_values, get_hsv_mask

LEAF_COLOR: str = "#00E500"
DATASET_PATH: Path = Path(
    "submission/datasets/dataset_level2_nr_seeds_6_nr_ite_100000.npy"
)
# color(--hsv 120 100% 89.804%)


# ================================================================
# 1. Section: Functions
# ================================================================
def obstacle_by_hue(
    image: np.ndarray,
    target_hue: float = 120,
    tolerance_hue: float = 5,
    min_saturation: float = 0.3,
    min_value: float = 0.8,
    height_threshold: int = 150,
) -> np.ndarray:
    # 1. Builds a hsv dependent mask (isolate bright leafs)
    mask = get_hsv_mask(
        image=image,
        target_hue=target_hue,
        tolerance_hue=tolerance_hue,
        min_saturation=min_saturation,
        min_value=min_value,
    )

    # 2. Extract the tall objects (closer)
    obstacle_centroids = get_obstacles_by_height(mask, height_threshold)

    # 3. Get the avoidance signals
    signals = get_signals_from_centroids(
        obstacle_centroids,
        np.asarray(image.shape)
    )

    return signals


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def is_flipped(
    image: np.ndarray,
    target_hue: float = 120,
    tolerance_hue: float = 5,
) -> bool:
    hsv_image = convert_to_hsv(image)
    hue, _, _ = get_hsv_values(hsv_image)

    target = hue_to_degree(target_hue)
    tolerance = hue_to_degree(tolerance_hue)

    # Circular hue distance
    diff = np.abs(hue - target)
    diff = np.minimum(diff, 1.0 - diff)

    # Saturation filter avoids meaningless hue values in gray/black/white areas
    mask = diff <= tolerance

    # If no pixels matched, cannot determine flipped state
    if not np.any(mask):
        return False

    ys, _ = np.where(mask)
    mask_center_y = ys.mean()
    image_center_y = image.shape[0] / 2

    return mask_center_y < image_center_y


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    dataset = np.load(DATASET_PATH)

    for img in dataset:
        if is_flipped(img):
            continue

        img = prepare_image_for_png(img)
        centroids = obstacle_by_hue(img)

        print(centroids)
        if centroids:
            break
