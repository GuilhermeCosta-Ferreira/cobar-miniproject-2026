# ================================================================
# 0. Section: IMPORTS
# ================================================================
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
