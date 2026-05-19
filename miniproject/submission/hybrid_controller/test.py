import numpy as np
from scipy.interpolate import interp1d

from flygym.examples.locomotion.cpg_network import CPGNetwork
from flygym.examples.locomotion.preprogrammed_steps import PreprogrammedSteps


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


class HybridTurningController:
    """CPG turning controller with hybrid sensory-feedback corrections.

    This is your CPG turning controller plus the NeuroMechFly hybrid rules:
    - retraction correction for legs stuck in holes
    - stumbling correction for tibia/Tarsus1/Tarsus2 collisions

    Unlike a pure CPG controller, this controller needs the current simulation
    observation in `step(...)`.

    Parameters
    ----------
    timestep : float
        Simulation timestep in seconds.
    contact_sensor_placements : list[str], optional
        Use `fly.contact_sensor_placements` after creating the Fly.
        Required for stumbling correction.
    """

    correction_vectors = {
        # "leg pos": (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1)
        "F": np.array([-0.03, 0.0, 0.0, -0.03, 0.0, 0.03, 0.03]),
        "M": np.array([-0.015, 0.001, 0.025, -0.02, 0.0, -0.02, 0.0]),
        "H": np.array([0.0, 0.0, 0.0, -0.02, 0.0, 0.01, -0.02]),
    }

    # Per-DoF sign inversion for right-side legs.
    right_leg_inversion = np.array([1, -1, -1, 1, -1, 1, 1])

    def __init__(
        self,
        timestep,
        contact_sensor_placements=None,
        intrinsic_freqs=np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 1,
        phase_biases=_tripod_phase_biases,
        coupling_weights=_tripod_coupling_weights,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        seed=0,
        stumbling_force_threshold=-1.0,
        correction_rates=None,
        max_correction=80.0,
        retraction_margin=0.05,
        retraction_persistence_steps=20,
        persistence_init_threshold=20.0,
        detected_segments=("Tibia", "Tarsus1", "Tarsus2"),
    ):
        self.timestep = timestep
        self.preprogrammed_steps = PreprogrammedSteps()

        self.base_intrinsic_freqs = np.asarray(intrinsic_freqs, dtype=float).copy()
        self.base_intrinsic_amps = np.asarray(intrinsic_amps, dtype=float).copy()

        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=self.base_intrinsic_freqs.copy(),
            intrinsic_amps=self.base_intrinsic_amps.copy(),
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            init_phases=init_phases,
            init_magnitudes=init_magnitudes,
            seed=seed,
        )

        self.stumbling_force_threshold = stumbling_force_threshold
        self.correction_rates = correction_rates or {
            "retraction": (800, 700),
            "stumbling": (2200, 1800),
        }
        self.max_correction = max_correction
        self.retraction_margin = retraction_margin
        self.retraction_persistence_steps = retraction_persistence_steps
        self.persistence_init_threshold = persistence_init_threshold
        self.detected_segments = tuple(detected_segments)

        self.step_phase_gain = self._make_step_phase_gain()

        self.stumbling_sensors = None
        if contact_sensor_placements is not None:
            self.set_contact_sensor_placements(contact_sensor_placements)

        self.reset(init_phases=init_phases, init_magnitudes=init_magnitudes)

    def _make_step_phase_gain(self):
        """Build phase-dependent correction gain used by the hybrid controller."""
        step_phase_gain = {}

        for leg in self.preprogrammed_steps.legs:
            swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]

            step_points = [
                swing_start,
                np.mean([swing_start, swing_end]),
                swing_end + np.pi / 4,
                np.mean([swing_end, 2 * np.pi]),
                2 * np.pi,
            ]
            increment_vals = [0, 0.8, 0, -0.1, 0]

            # Match the official hybrid controller: extend the swing interval.
            self.preprogrammed_steps.swing_period[leg] = (
                swing_start,
                swing_end + np.pi / 4,
            )

            step_phase_gain[leg] = interp1d(
                step_points,
                increment_vals,
                kind="linear",
                fill_value="extrapolate",
            )

        return step_phase_gain

    def set_contact_sensor_placements(self, contact_sensor_placements):
        """Map contact sensor indices to legs for stumbling detection.

        Call this with `fly.contact_sensor_placements` after constructing `Fly`.
        """
        stumbling_sensors = {leg: [] for leg in self.preprogrammed_steps.legs}

        for i, sensor_name in enumerate(contact_sensor_placements):
            # Official examples often look like "Animat/LFTarsus1";
            # user-provided placements may look like "LFTarsus1".
            body_name = sensor_name.split("/")[-1]
            leg = body_name[:2]
            segment = body_name[2:]

            if leg in stumbling_sensors and segment in self.detected_segments:
                stumbling_sensors[leg].append(i)

        self.stumbling_sensors = {
            leg: np.asarray(indices, dtype=int)
            for leg, indices in stumbling_sensors.items()
        }

        missing = [
            leg
            for leg, indices in self.stumbling_sensors.items()
            if len(indices) != len(self.detected_segments)
        ]
        if missing:
            raise RuntimeError(
                "Missing stumbling contact sensors for legs "
                f"{missing}. Enable contact sensors for: {self.detected_segments}."
            )

    def reset(self, init_phases=None, init_magnitudes=None):
        self.cpg_network.reset(
            init_phases=init_phases,
            init_magnitudes=init_magnitudes,
        )

        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)
        self.retraction_persistence_counter = np.zeros(6)

        self.last_net_corrections = np.zeros(6)
        self.last_retraction_leg = None

    def _update_turning_cpg(self, action):
        """Same descending-signal turning modulation as your CPG controller."""
        action = np.asarray(action, dtype=float)
        if action.shape != (2,):
            raise ValueError(f"`action` must have shape (2,), got {action.shape}.")

        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()

        freqs = self.base_intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1

        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

    def _find_retraction_leg(self, obs):
        """Return the leg index that appears stuck in a hole, or None."""
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]

        sorted_idx = np.argsort(end_effector_z_pos)
        sorted_z = end_effector_z_pos[sorted_idx]

        if sorted_z[-1] > sorted_z[-3] + self.retraction_margin:
            leg_idx = sorted_idx[-1]

            if self.retraction_correction[leg_idx] > self.persistence_init_threshold:
                self.retraction_persistence_counter[leg_idx] = 1

            return leg_idx

        return None

    def step(self, action, obs):
        """Step controller forward.

        Parameters
        ----------
        action : np.ndarray
            Shape (2,), descending signal [delta_L, delta_R].
            Use np.array([1.0, 1.0]) for straight walking.
        obs : dict
            Current NeuroMechFly observation from sim.reset() or sim.step(...).

        Returns
        -------
        joint_angles : np.ndarray
            Flattened joint angles, shape (42,).
        adhesion : np.ndarray
            Adhesion on/off signal per leg, shape (6,).
        """
        if self.stumbling_sensors is None:
            raise RuntimeError(
                "Stumbling sensors are not initialized. Pass "
                "`contact_sensor_placements=fly.contact_sensor_placements` "
                "when constructing the controller, or call "
                "`controller.set_contact_sensor_placements(...)`."
            )

        self._update_turning_cpg(action)

        # Retraction rule: does a leg need to be retracted from a hole?
        leg_to_correct_retraction = self._find_retraction_leg(obs)
        self.last_retraction_leg = leg_to_correct_retraction

        # Update persistence counter.
        active = self.retraction_persistence_counter > 0
        self.retraction_persistence_counter[active] += 1
        self.retraction_persistence_counter[
            self.retraction_persistence_counter > self.retraction_persistence_steps
        ] = 0

        self.cpg_network.step()

        joint_angles = np.zeros((6, 7))
        adhesion = np.zeros(6, dtype=int)
        all_net_corrections = np.zeros(6)

        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # --- Retraction correction ---
            if (
                i == leg_to_correct_retraction
                or self.retraction_persistence_counter[i] > 0
            ):
                increment = self.correction_rates["retraction"][0] * self.timestep
                self.retraction_correction[i] += increment
            else:
                decrement = self.correction_rates["retraction"][1] * self.timestep
                self.retraction_correction[i] = max(
                    0.0,
                    self.retraction_correction[i] - decrement,
                )

            # --- Stumbling correction ---
            contact_forces = obs["contact_forces"][self.stumbling_sensors[leg], :]
            fly_orientation = obs["fly_orientation"]

            # Negative projection means force against heading.
            force_proj = np.dot(contact_forces, fly_orientation)

            if (force_proj < self.stumbling_force_threshold).any():
                increment = self.correction_rates["stumbling"][0] * self.timestep
                self.stumbling_correction[i] += increment
            else:
                decrement = self.correction_rates["stumbling"][1] * self.timestep
                self.stumbling_correction[i] = max(
                    0.0,
                    self.stumbling_correction[i] - decrement,
                )

            # Retraction has priority over stumbling.
            if self.retraction_correction[i] > 0:
                net_correction = self.retraction_correction[i]
                self.stumbling_correction[i] = 0.0
            else:
                net_correction = self.stumbling_correction[i]

            # Base CPG joint angles.
            phase = self.cpg_network.curr_phases[i]
            magnitude = self.cpg_network.curr_magnitudes[i]

            my_joint_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                phase,
                magnitude,
            )

            # Clip and apply phase-dependent gain.
            net_correction = np.clip(net_correction, 0.0, self.max_correction)
            net_correction *= self.step_phase_gain[leg](phase % (2 * np.pi))

            # Apply leg-specific correction vector.
            correction_vector = self.correction_vectors[leg[1]].copy()

            # Mirror correction for right-side legs.
            if leg[0] == "R":
                correction_vector *= self.right_leg_inversion

            my_joint_angles += net_correction * correction_vector

            joint_angles[i] = my_joint_angles
            adhesion[i] = self.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            all_net_corrections[i] = net_correction

        self.last_net_corrections = all_net_corrections

        return joint_angles.ravel(), adhesion
