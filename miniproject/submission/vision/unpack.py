# ================================================================
# 0. Section: IMPORTS
# ================================================================
import cv2

import numpy as np

from pathlib import Path


# ================================================================
# 1. Section: Functions
# ================================================================
def unpack_dataset(dataset_path: Path) -> Path:
    folder = dataset_path.parent / dataset_path.stem
    folder.mkdir(parents=True, exist_ok=True)

    dataset = np.load(dataset_path)

    for i, img in enumerate(dataset):
        img = prepare_image_for_png(img)

        out_file = folder / f"img_{i:06d}.png"
        cv2.imwrite(str(out_file), img)

    print(f"Saved {len(dataset)} images to {folder.resolve()}")
    return folder


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def prepare_image_for_png(img: np.ndarray) -> np.ndarray:
    """
    Converts one dataset entry into a format cv2.imwrite can save.
    """

    img = np.asarray(img)

    # If image is float in [0, 1], convert to uint8 [0, 255]
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

    # If image is integer but not uint8, clip and convert
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # If RGB image, convert RGB -> BGR for OpenCV
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # If RGBA image, convert RGBA -> BGRA
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    return img


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    unpack_dataset(
        Path("submission/datasets/failed_trial_2_nr_seeds_1_nr_ite_50000.npy")
    )
