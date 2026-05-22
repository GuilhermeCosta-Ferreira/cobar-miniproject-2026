# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt
import cv2

from pathlib import Path

from .unpack import prepare_image_for_png
from .obstacles import get_obstacles_by_height_fast, get_signals_from_centroids
from .hsv import convert_to_hsv, hue_to_degree, get_hsv_values, get_hsv_mask

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
    turn_gain: float = 1.5,
) -> np.ndarray:
    # 1. Builds a hsv dependent mask (isolate bright leafs)
    mask = get_hsv_mask(
        image=image,
        target_hue=target_hue,
        tolerance_hue=tolerance_hue,
        target_saturation=min_saturation,
        target_value=min_value,
    )

    # 2. Extract the tall objects (closer)
    obstacle_centroids = get_obstacles_by_height_fast(mask, height_threshold)

    # 3. Get the avoidance signals
    signals = get_signals_from_centroids(
        obstacle_centroids,
        np.asarray(image.shape),
        turn_gain=turn_gain,
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


def stack_raw_vision(
    raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
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


def dragonfly_red_scores_by_eye(
    raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
    r_min: float = 90,
    dominance: float = 1.25,
    red_minus_green_min: float = 20,
) -> tuple[float, float, float]:
    """
    Compute red score for each eye separately.

    Returns
    -------
    total_score:
        Mean red score over all eye pixels.
    left_score:
        Red score in the first raw vision image.
    right_score:
        Red score in the second raw vision image.
    """
    if len(raw_vision) < 2:
        total_score, _ = dragonfly_red_score_from_raw_vision(
            raw_vision,
            r_min=r_min,
            dominance=dominance,
            red_minus_green_min=red_minus_green_min,
        )
        return total_score, total_score, total_score

    left_score, _ = red_score_from_rgb(
        raw_vision[0],
        r_min=r_min,
        dominance=dominance,
        red_minus_green_min=red_minus_green_min,
    )
    right_score, _ = red_score_from_rgb(
        raw_vision[1],
        r_min=r_min,
        dominance=dominance,
        red_minus_green_min=red_minus_green_min,
    )
    total_score = 0.5 * (left_score + right_score)

    return total_score, left_score, right_score


def dragonfly_red_features_from_raw_vision(
    raw_vision: list[np.ndarray] | tuple[np.ndarray, ...],
    r_min: float = 90,
    dominance: float = 1.25,
    red_minus_green_min: float = 60,
    hsv_sat_min: int = 80,
    hsv_val_min: int = 70,
) -> dict[str, float]:
    """
    Extract red-head features from the fly's raw vision.

    """
    raw_panel = stack_raw_vision(raw_vision)
    img = to_uint8_rgb(raw_panel)
    rgb = img.astype(np.float32)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    dominance_mask = (
        (r > r_min)
        & (r > dominance * g)
        & (r > dominance * b)
        & ((r - g) > red_minus_green_min)
    )

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    hsv_mask = (
        ((hue <= 10) | (hue >= 170)) & (sat >= hsv_sat_min) & (val >= hsv_val_min)
    )

    red_mask = dominance_mask | hsv_mask
    mask = np.ascontiguousarray(red_mask.astype(np.uint8))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    panel_area = float(mask.shape[0] * mask.shape[1])
    largest_blob_frac = 0.0
    blob_x = 0.5
    blob_y = 0.5
    component_count = max(0, n_labels - 1)

    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        label_id = int(np.argmax(areas)) + 1
        largest_blob_frac = float(stats[label_id, cv2.CC_STAT_AREA]) / panel_area
        cx, cy = centroids[label_id]
        blob_x = float(cx) / mask.shape[1]
        blob_y = float(cy) / mask.shape[0]

    if len(raw_vision) >= 2:
        mid_col = mask.shape[1] // 2
        left_score = float(mask[:, :mid_col].mean())
        right_score = float(mask[:, mid_col:].mean())
    else:
        left_score = float(mask.mean())
        right_score = left_score

    score_sum = left_score + right_score
    side_bias = 0.0 if score_sum == 0 else (right_score - left_score) / score_sum

    return {
        "red_score": float(mask.mean()),
        "dominance_score": float(dominance_mask.mean()),
        "hsv_red_score": float(hsv_mask.mean()),
        "left_score": left_score,
        "right_score": right_score,
        "side_bias": float(side_bias),
        "largest_blob_frac": largest_blob_frac,
        "blob_x": blob_x,
        "blob_y": blob_y,
        "component_count": float(component_count),
    }


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
            tolerance=5,
            target_saturation=0.3,
            target_value=0.8,
        )

        obstacle_centroids = get_obstacles_by_height_fast(mask, 100)

        print()
        print(obstacle_centroids)
        print()

        signals = get_signals_from_centroids(
            obstacle_centroids, np.asarray(img.shape), turn_gain=1
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
