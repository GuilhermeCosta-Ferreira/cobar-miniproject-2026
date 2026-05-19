# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scipy.interpolate import interp1d

from flygym.examples.locomotion.cpg_network import CPGNetwork
from flygym.examples.locomotion.preprogrammed_steps import PreprogrammedSteps



# ================================================================
# 1. Section: INPUTS
# ================================================================
_tripod_phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
_tripod_coupling_weights = (_tripod_phase_biases > 0) * 10
_correction_vectors = {
    # "leg pos": (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1)
    # unit: radian
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}
_right_leg_inversion = [1, -1, -1, 1, -1, 1, 1]
_stumbling_force_threshold = -1
_correction_rates: dict[str, tuple[int, int]] = {"retraction": (800, 700), "stumbling": (2200, 1800)}
_max_increment = 80
_retraction_persistance = 20
_persistance_init_thr = 20




# ================================================================
# 2. Section: MAIN CLASS
# ================================================================
class HybridController:
    def __init__(
        self,
        timestep: float,
        intrinsic_freqs: np.ndarray = np.ones(6) * 12,
        intrinsic_amps: np.ndarray = np.ones(6) * 1,
        phase_biases: np.ndarray = _tripod_phase_biases,
        coupling_weights: np.ndarray = _tripod_coupling_weights,
        convergence_coefs: np.ndarray = np.ones(6) * 20,
        correction_rates: dict[str, tuple[int, int]] = _correction_rates,
        stumbling_force_threshold: float | int = _stumbling_force_threshold,
        stumbling_correction: np.ndarray = np.zeros(6),
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
        seed: int = 0,
    ):
        self.preprogrammed_steps = PreprogrammedSteps()
        self.intrinsic_freqs = intrinsic_freqs
        self.timestep = timestep
        self.correction_rates = correction_rates
        self.stumbling_force_threshold = stumbling_force_threshold
        self.stumbling_correction = stumbling_correction
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            init_phases=init_phases,
            init_magnitudes=init_magnitudes,
            seed=seed,
        )

    def reset(self, init_phases=None, init_magnitudes=None):
        self.cpg_network.reset(init_phases=init_phases, init_magnitudes=init_magnitudes)

    def step(self, sim, action):
        """Step the controller forward one timestep.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing the descending signal
            [delta_L, delta_R] for turning.

        Returns
        -------
        joint_angles : np.ndarray
            Flattened array of joint angles, shape (42,).
        adhesion : np.ndarray
            Adhesion on/off signal per leg, shape (6,).
        """
        step_phase_multipler = {}
        for leg in self.preprogrammed_steps.legs:
            swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]

            step_points = [
                swing_start,
                np.mean([swing_start, swing_end]),
                swing_end + np.pi / 4,
                np.mean([swing_end, 2 * np.pi]),
                2 * np.pi,
            ]
            self.preprogrammed_steps.swing_period[leg] = (swing_start, swing_end + np.pi / 4)
            increment_vals = [0, 0.8, 0, -0.1, 0]

            step_phase_multipler[leg] = interp1d(
                step_points, increment_vals, kind="linear", fill_value="extrapolate"
            )

        retraction_correction = np.zeros(6)

        # contact_forces_by_leg shape: (6, 3)
        # Must be ordered the same way as preprogrammed_steps.legs:
        # usually ["LF", "LM", "LH", "RF", "RM", "RH"]

        contact_forces_by_leg = get_contact_forces(sim)
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            contact_force = contact_forces_by_leg[i]
            fly_orientation = get_fly_orientation(sim)

            # Negative projection means force against the fly's heading.
            force_proj = np.dot(contact_force, fly_orientation)

            if force_proj < self.stumbling_force_threshold:
                increment = self.correction_rates["stumbling"][0] * self.timestep
                self.stumbling_correction[i] += increment
            else:
                decrement = self.correction_rates["stumbling"][1] * self.timestep
                self.stumbling_correction[i] = max(
                    0.0,
                    self.stumbling_correction[i] - decrement,
                )


        # Update CPG parameters based on descending signal
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        self.cpg_network.step()

        joint_angles = np.zeros((6, 7))
        adhesion = np.zeros(6)
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            joint_angles[i] = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            adhesion[i] = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )

        return joint_angles.ravel(), adhesion


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def quat_to_forward_vector(q) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to world-frame body +x axis."""
    w, x, y, z = q

    # Rotation matrix, first column = rotated +x axis.
    forward = np.array([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ])

    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])

    return forward / norm

def get_fly_orientation(sim) -> np.ndarray:
    fly_name = sim.fly.name
    body_rotations = sim.get_body_rotations(fly_name)
    bodyseg_order = sim.fly.get_bodysegs_order()
    bodyseg_names = [
            seg.name if hasattr(seg, "name") else str(seg)
            for seg in bodyseg_order
        ]
    root_name = sim.fly.root_segment.name if hasattr(sim.fly.root_segment, "name") else str(sim.fly.root_segment)
    root_idx = bodyseg_names.index(root_name)
    root_quat = body_rotations[root_idx]
    return quat_to_forward_vector(root_quat)

def _seg_name(seg):
    return seg.name if hasattr(seg, "name") else str(seg)

def get_contact_forces(sim) -> np.ndarray:
    tarsus5_names = [
        "lf_tarsus5",
        "lm_tarsus5",
        "lh_tarsus5",
        "rf_tarsus5",
        "rm_tarsus5",
        "rh_tarsus5",
    ]
    external_forces = sim.get_external_force(
        sim.fly.fly_name,
        subtract_adhesion_force=True,
    )
    contact_names = [_seg_name(k) for k in sim.fly.contactbodyseg_to_mjcfbody.keys()]
    force_by_contact_name = {
        name: external_forces[i]
        for i, name in enumerate(contact_names)
    }
    return np.stack([
        force_by_contact_name[name]
        for name in tarsus5_names
    ])
