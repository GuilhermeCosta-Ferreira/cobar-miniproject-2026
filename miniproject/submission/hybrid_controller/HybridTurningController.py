# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scipy.interpolate import interp1d

from flygym.examples.locomotion.cpg_network import CPGNetwork
from flygym.examples.locomotion.preprogrammed_steps import PreprogrammedSteps
from miniproject.simulation import MiniprojectSimulation

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
"""
_correction_vectors = {
    # "leg pos": (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1)
    # unit: radian
    "f": np.array([0, 0, 0, 0, 0, 0, 0]),
    "m": np.array([0, 0, 0, 0, 0, 0, 0]),
    "h": np.array([0, 0, 0, 0, 0, 0, 0]),
}
_correction_vectors = {
    "f": np.array([0.0, 0.0, 0.0, 0.004, 0.0, -0.004, 0.0]),
    "m": np.array([0.0, 0.0, 0.0, 0.004, 0.0, -0.004, 0.0]),
    "h": np.array([0.0, 0.0, 0.0, 0.004, 0.0, -0.004, 0.0]),
}
"""
_correction_vectors = {
    "f": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "m": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "h": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}

_right_leg_inversion = [1, -1, -1, 1, -1, 1, 1]
_stumbling_force_threshold = -1
_correction_rates: dict[str, tuple[int, int]] = {
    "retraction": (800, 700),
    "stumbling": (2200, 1800),
}
_max_increment = 80
_retraction_persistance = 20
_persistance_init_thr = 20


# ================================================================
# 1. Section: Functions
# ================================================================
class HybridTurningController:
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
        persistance_init_thr: int | float = _persistance_init_thr,
        retraction_perisitance_counter=np.zeros(6),
        retraction_persistance: int | float = _retraction_persistance,
        run_time: float = 1.0,
        max_increment: int | float = _max_increment,
        right_leg_inversion: list[int] = _right_leg_inversion,
        correction_vectors: dict = _correction_vectors,
        init_phases: np.ndarray | None = None,
        init_magnitudes: np.ndarray | None = None,
        seed: int = 0,
    ):
        self.preprogrammed_steps = PreprogrammedSteps()
        self.intrinsic_freqs = intrinsic_freqs
        self.timestep = 1e-4
        self.correction_rates = correction_rates
        self.stumbling_force_threshold = stumbling_force_threshold
        self.stumbling_correction = stumbling_correction
        self.retraction_correction = np.zeros(6)
        self.persistance_init_thr = persistance_init_thr
        self.retraction_perisitance_counter = retraction_perisitance_counter
        self.retraction_persistance = retraction_persistance
        self.max_increment = max_increment
        self.right_leg_inversion = right_leg_inversion
        self.correction_vectors = correction_vectors

        target_num_steps = int(run_time / timestep)
        self.retraction_persistance_counter_hist = np.zeros((6, target_num_steps))

        self.contact_sensor_placements = build_contact_sensor_placements(
            self.preprogrammed_steps
        )
        self.step_phase_multipler = build_step_phase_initializer(
            self.preprogrammed_steps
        )
        self.stumbling_sensors = build_stumbling_sensors(
            self.preprogrammed_steps, self.contact_sensor_placements
        )

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

        self.legs_diff_to_body = []
        self.force_hist = {}
        self.corrected_leg = []

    def step(
        self, sim: MiniprojectSimulation, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        # 1. Get the leg that needs retraction (or None)
        leg_to_correct_retraction = self.retraction_rule(sim)
        self.corrected_leg.append(leg_to_correct_retraction)

        # 2. Update the retraction persistance
        self.retraction_perisitance_counter[
            self.retraction_perisitance_counter > 0
        ] += 1
        self.retraction_perisitance_counter[
            self.retraction_perisitance_counter > self.retraction_persistance
        ] = 0
        # self.retraction_persistance_counter_hist[:, sim._curr_step] = self.retraction_perisitance_counter

        # 3. Run the cpg
        self.cpg_network.step()
        joints_angles = []
        adhesion_onoff = []
        all_net_corrections = np.zeros(6)

        # 4. Check each leg
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # 4.1 Update amount of retraction correction
            self.update_retraction_correction(i, sim, leg_to_correct_retraction)

            # 4.2 Update amount of stumpling correction
            self.update_stumbling_correction(i, leg, sim)

            # 4.3 Retraction correction is prioritized
            if self.retraction_correction[i] > 0:
                net_correction = self.retraction_correction[i]
                self.stumbling_correction[i] = 0
            else:
                net_correction = self.stumbling_correction[i]

            # 4.4 Get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            net_correction = np.clip(net_correction, 0, self.max_increment)
            phase_gain = self.step_phase_multipler[leg](
                self.cpg_network.curr_phases[i] % (2 * np.pi)
            )

            correction_vector = self.correction_vectors[leg[1]].copy()

            if leg[0] == "r":
                correction_vector *= np.asarray(self.right_leg_inversion)

            my_joints_angles += net_correction * phase_gain * correction_vector
            joints_angles.append(my_joints_angles)

            # 4.5 Get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )
            all_net_corrections[i] = net_correction
            adhesion_onoff.append(my_adhesion_onoff)

        return np.array(np.concatenate(joints_angles)), np.array(adhesion_onoff)

    def retraction_rule(
        self,
        sim: MiniprojectSimulation,
        body_reference: str = "c_thorax",
    ) -> int | None:
        """Returns the leg that meeds retraction (or just None)"""
        # 0. Get the z positions
        body_z = get_bodypart_pos(sim, body_reference)[2]
        legs_z_pos = get_legs_pos(sim, list(self.stumbling_sensors.keys()))[:, 2]

        # 1. Compute how low each leg is compared to the body
        end_effector_z_pos = body_z - legs_z_pos
        self.legs_diff_to_body.append(end_effector_z_pos)

        # 2. Sort the legs by the relative drop
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)

        # 3. Get the sorted values
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]

        # 4. Check if the lowest leg is an outlier
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.5:
            # 4.1. Chose the leg to correct
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]

            # 4.2. Activate persistence if correction is already strong
            if (
                self.retraction_correction[leg_to_correct_retraction]
                > self.persistance_init_thr
            ):
                self.retraction_perisitance_counter[leg_to_correct_retraction] = 1
        # 7. Otherwise, no leg needs retraction
        else:
            leg_to_correct_retraction = None

        return leg_to_correct_retraction

    def update_retraction_correction(
        self,
        leg_idx: int,
        sim: MiniprojectSimulation,
        leg_to_correct_retraction: int | None,
    ) -> None:
        if (
            leg_idx == leg_to_correct_retraction
            or self.retraction_perisitance_counter[leg_idx] > 0
        ):  # lift leg
            increment = self.correction_rates["retraction"][0] * sim.timestep
            self.retraction_correction[leg_idx] += increment
        else:  # condition no longer met, lower leg
            decrement = self.correction_rates["retraction"][1] * sim.timestep
            self.retraction_correction[leg_idx] = max(
                0, self.retraction_correction[leg_idx] - decrement
            )

    def update_stumbling_correction(
        self,
        leg_idx: int,
        leg: str,
        sim: MiniprojectSimulation,
    ) -> None:
        contact_forces = sim.get_external_force(sim.fly.name, True)[
            self.stumbling_sensors[leg], :
        ]
        fly_orientation = get_fly_orientation(sim)
        if leg not in self.force_hist:
            self.force_hist[leg] = []
        self.force_hist[leg].append(contact_forces)

        # force projection should be negative if against fly orientation
        force_proj = np.dot(contact_forces, fly_orientation)
        if (force_proj < self.stumbling_force_threshold).any():
            increment = self.correction_rates["stumbling"][0] * sim.timestep
            self.stumbling_correction[leg_idx] += increment
        else:
            decrement = self.correction_rates["stumbling"][1] * sim.timestep
            self.stumbling_correction[leg_idx] = max(
                0, self.stumbling_correction[leg_idx] - decrement
            )


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def build_contact_sensor_placements(
    preprogrammed_steps: PreprogrammedSteps,
) -> list[str]:
    return [
        f"{leg}_{segment}".lower()
        for leg in preprogrammed_steps.legs
        for segment in ["tarsus5"]
    ]


def build_step_phase_initializer(preprogrammed_steps: PreprogrammedSteps) -> dict:
    step_phase_multipler = {}

    for leg in preprogrammed_steps.legs:
        swing_start, swing_end = preprogrammed_steps.swing_period[leg]

        step_points = [
            swing_start,
            np.mean([swing_start, swing_end]),
            swing_end + np.pi / 4,
            np.mean([swing_end, 2 * np.pi]),
            2 * np.pi,
        ]
        preprogrammed_steps.swing_period[leg] = (swing_start, swing_end + np.pi / 4)
        increment_vals = [0, 0.8, 0, -0.1, 0]

        step_phase_multipler[leg] = interp1d(
            step_points, increment_vals, kind="linear", fill_value="extrapolate"
        )

    return step_phase_multipler


def build_stumbling_sensors(
    preprogrammed_steps: PreprogrammedSteps, contact_sensor_placements: list[str]
) -> dict:
    detected_segments = ["tarsus5"]
    stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}
    for i, sensor_name in enumerate(contact_sensor_placements):
        leg = sensor_name.rsplit("_")[0]  # sensor_name: eg. "Animat/LFTarsus1"
        segment = sensor_name.rsplit("_")[1]
        if segment in detected_segments:
            stumbling_sensors[leg].append(i)
    return {k: np.array(v) for k, v in stumbling_sensors.items()}


def get_bodypart_pos(sim: MiniprojectSimulation, bodypart: str) -> np.ndarray:
    for idx, bs in enumerate(sim.fly.get_bodysegs_order()):
        if bs.name == bodypart:
            return sim.get_body_positions(sim.fly.name)[idx]

    raise ValueError(f"No bodypart with name {bodypart} found in {sim.fly.name}")


def get_legs_pos(sim: MiniprojectSimulation, legs: list[str]) -> np.ndarray:
    legs_pos = []
    for leg in legs:
        legs_pos.append(get_bodypart_pos(sim, f"{leg}_tibia"))

    return np.asarray(legs_pos)


def get_fly_orientation(sim: MiniprojectSimulation) -> np.ndarray:
    fly_name = sim.fly.name
    body_rotations = sim.get_body_rotations(fly_name)
    bodyseg_order = sim.fly.get_bodysegs_order()
    bodyseg_names = [
        seg.name if hasattr(seg, "name") else str(seg) for seg in bodyseg_order
    ]
    root_name = (
        sim.fly.root_segment.name
        if hasattr(sim.fly.root_segment, "name")
        else str(sim.fly.root_segment)
    )
    root_idx = bodyseg_names.index(root_name)
    root_quat = body_rotations[root_idx]
    return quat_to_forward_vector(root_quat)


def quat_to_forward_vector(q: np.ndarray | list) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to world-frame body +x axis."""
    w, x, y, z = q

    # Rotation matrix, first column = rotated +x axis.
    forward = np.array(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
        ]
    )

    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])

    return forward / norm
