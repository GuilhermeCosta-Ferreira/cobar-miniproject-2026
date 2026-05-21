# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path

from .unpack import prepare_image_for_png
from .obstacles import get_obstacles_by_height_fast, get_signals_from_centroids
from .hsv import convert_to_hsv, hue_to_degree, get_hsv_values, get_hsv_mask_fast

LEAF_COLOR: str = "#00E500"
DATASET_PATH: Path = Path(
    "submission/datasets/failed_trial_nr_seeds_1_nr_ite_50000.npy"
)
# target color(--hsv 120 100% 89.804%)


# ================================================================
# 1. Section: Functions
# ================================================================
def obstacle_by_hue(
    image: np.ndarray,
    target_hue: float = 120,
    tolerance_hue: float = 5,
    min_saturation: float = 0.3,
    min_value: float = 0.8,
    height_threshold: int = 100,
) -> np.ndarray:
    # 1. Builds a hsv dependent mask (isolate bright leafs)
    mask = get_hsv_mask_fast(
        image=image,
        target_hue=target_hue,
        tolerance_hue=tolerance_hue,
        min_saturation=min_saturation,
        min_value=min_value,
    )

    # 2. Extract the tall objects (closer)
    obstacle_centroids = get_obstacles_by_height_fast(mask, height_threshold)

    # 3. Get the avoidance signals
    signals = get_signals_from_centroids(
        obstacle_centroids,
        np.asarray(image.shape),
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

# ──────────────────────────────────────────────────────
# Dragonfly attack detection
# ──────────────────────────────────────────────────────

def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB uint8.
    Handles images in [0, 1] or [0, 255].
    """
    img = np.asarray(img)

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def stack_raw_vision(raw_vision: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Concatenate the fly's raw vision images into one panel.

    """
    eyes = [to_uint8_rgb(eye) for eye in raw_vision]
    return np.concatenate(eyes, axis=1)


def red_score_from_rgb(
    img_rgb: np.ndarray,
    r_min: float = 90,
    dominance: float = 1.25,
    red_minus_green_min: float = 20,
) -> tuple[float, np.ndarray]:
    """
    Detect red pixels in an RGB image.

    Returns
    -------
    score:
        Fraction of pixels classified as red.
    mask:
        Boolean mask of red pixels.
    """
    img = to_uint8_rgb(img_rgb).astype(np.float32)

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    mask = (
        (r > r_min)
        & (r > dominance * g)
        & (r > dominance * b)
        & ((r - g) > red_minus_green_min)
    )

    score = float(mask.mean())
    return score, mask


def dragonfly_red_score_from_raw_vision(
    raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
    r_min: float = 90,
    dominance: float = 1.25,
    red_minus_green_min: float = 20,
) -> tuple[float, np.ndarray]:
    """
    Compute dragonfly red score directly from raw fly vision.
    """
    raw_panel = stack_raw_vision(raw_vision)

    score, mask = red_score_from_rgb(
        raw_panel,
        r_min=r_min,
        dominance=dominance,
        red_minus_green_min=red_minus_green_min,
    )

    return score, mask


def detect_dragonfly_attack_from_raw_vision(
    raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
    attack_threshold: float = 0.06,
    r_min: float = 90,
    dominance: float = 1.25,
    red_minus_green_min: float = 20,
) -> tuple[bool, float]:
    """
    Detect whether the dragonfly is attacking based on the red-head cue.

    """
    score, _ = dragonfly_red_score_from_raw_vision(
        raw_vision,
        r_min=r_min,
        dominance=dominance,
        red_minus_green_min=red_minus_green_min,
    )

    attack_detected = score > attack_threshold

    return attack_detected, score





# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    dataset = np.load(DATASET_PATH)

    for img in dataset:
        if is_flipped(img):
            continue

        mask = get_hsv_mask_fast(
            image=img,
            target_hue=120,
            tolerance_hue=5,
            min_saturation=0.3,
            min_value=0.8,
        )

        obstacle_centroids = get_obstacles_by_height_fast(mask, 100)

        print()
        print(obstacle_centroids)
        print()

        signals = get_signals_from_centroids(
            obstacle_centroids,
            np.asarray(img.shape),
        )

        img = prepare_image_for_png(img)
        fig, axs = plt.subplots(1, 2, figsize=(14, 8))

        # Original image
        axs[0].imshow(img)
        for obs in obstacle_centroids:
            axs[0].scatter(obs[0], obs[1], s=30)
        axs[0].set_title("Image + centroids")
        axs[0].axis("off")

        # Mask
        axs[1].imshow(mask, cmap="gray")
        for obs in obstacle_centroids:
            axs[1].scatter(obs[0], obs[1], s=30)
        axs[1].set_title("Mask")
        axs[1].axis("off")

        fig.suptitle(f"signals = {signals}")
        plt.tight_layout()
        plt.show()
