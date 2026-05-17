# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path
from joblib import Parallel, delayed
from numpy.random import RandomState

from .build_train import run_simulation

# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    random_state = RandomState(0)
    output_basedir = Path("outputs/head_stabilization/random_exploration/")

    job_specs = []
    for gait in ["tripod", "tetrapod", "wave"]:
        for terrain in ["flat", "blocks"]:
            for test_set in [True, False]:
                # Get DN drives
                if test_set:
                    turning_drives = np.linspace(-0.9, 0.9, 10)
                else:
                    turning_drives = np.linspace(-1, 1, 11)  # staggered from test set
                amp_lower = np.maximum(1 - 0.6 * np.abs(turning_drives), 0.4)
                amp_upper = np.minimum(1 + 0.2 * np.abs(turning_drives), 1.2)
                dn_drives_left = np.where(turning_drives > 0, amp_upper, amp_lower)
                dn_drives_right = np.where(turning_drives > 0, amp_lower, amp_upper)

                set_tag = "test_set" if test_set else "train_set"
                for dn_left, dn_right in zip(dn_drives_left, dn_drives_right):
                    spawn_xy = random_state.uniform(-1.3, 1.3, size=2)
                    dn_drive = np.array([dn_left, dn_right])
                    output_dir = (
                        output_basedir
                        / f"{gait}_{terrain}_{set_tag}_{dn_left:.2f}_{dn_right:.2f}"
                    )
                    job_specs.append(
                        (gait, terrain, spawn_xy, dn_drive, 1.5, False, output_dir)
                    )

    Parallel(n_jobs=-2)(delayed(run_simulation)(*job_spec) for job_spec in job_specs)
