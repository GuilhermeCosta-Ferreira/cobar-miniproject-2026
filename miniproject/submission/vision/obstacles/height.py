# ================================================================
# 0. Section: IMPORTS
# ================================================================
import cv2

import numpy as np

from typing import cast

from scipy.ndimage import find_objects, center_of_mass
from scipy.ndimage import label as get_labels


# ================================================================
# 1. Section: Functions
# ================================================================
def get_obstacles_by_height(mask: np.ndarray, height_threshold: float) -> list:
    # 1. get the leafes
    labels, _ = cast(tuple, get_labels(mask))
    objects = find_objects(labels)

    valid_label_ids = []
    valid_centroids = []

    for label_id, obj_slice in enumerate(objects, start=1):
        if obj_slice is None:
            continue

        # Get the object height
        y_slice, _ = obj_slice
        height = y_slice.stop - y_slice.start

        # Get only the big objects
        if height >= height_threshold:
            cy, cx = center_of_mass(mask, labels, label_id)
            valid_label_ids.append(label_id)
            valid_centroids.append((cx, cy))

    return valid_centroids


def get_obstacles_by_height_fast(
    mask: np.ndarray,
    height_threshold: float,
) -> list[tuple[float, float]]:
    # OpenCV wants uint8 single-channel image.
    if mask.dtype == np.bool_:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask

    mask_u8 = np.ascontiguousarray(mask_u8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    valid_centroids: list[tuple[float, float]] = []

    # Label 0 is background, so start at 1.
    for label_id in range(1, num_labels):
        height = stats[label_id, cv2.CC_STAT_HEIGHT]

        if height >= height_threshold:
            cx, cy = centroids[label_id]
            valid_centroids.append((float(cx), float(cy)))

    return valid_centroids
