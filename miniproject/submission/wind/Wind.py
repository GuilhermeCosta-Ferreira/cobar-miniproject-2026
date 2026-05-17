# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from .quaternions import (
    mujoco_to_numpy_quaternion,
    numpy_to_mujoco_quaternion,
    quat_mul,
    quat_inv,
    quat_rotate,
)


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Wind:
    def __init__(self, model):
        # We need to store the rest pose of the antenna to be able to compute the deflection caused by the wind.
        # Since the pedicel is between the funiculus and the thorax in the kinematic chain, we need it to convert the funiculus deflection to the fly's reference frame.
        # The pedicel has no joints, so we can cache its position in the rest pose and use it as a reference for the funiculus deflection.
        self.funiculus_position_0 = [
            model.body_pos(model.body_name2id("funiculus_l")),
            model.body_pos(model.body_name2id("funiculus_r")),
        ]
        self.pedicel_angle_0 = [
            model.body_quat(model.body_name2id("pedicel_l")),
            model.body_quat(model.body_name2id("pedicel_r")),
        ]

    def _funiculus_position(self, antenna_data):
        qpos_l = antenna_data["l"]["qpos"]
        qpos_r = antenna_data["r"]["qpos"]

        funiculus_position_l = quat_rotate(qpos_l, self.funiculus_position_0[0])
        funiculus_position_r = quat_rotate(qpos_r, self.funiculus_position_0[1])

        return [funiculus_position_l, funiculus_position_r]
    
    def _compute_deflection(self, funiculus_position):
        # The deflection is the difference between the current funiculus position and the rest position.
        deflection_l = funiculus_position[0] - self.funiculus_position_0[0]
        deflection_r = funiculus_position[1] - self.funiculus_position_0[1]

        return [deflection_l, deflection_r]
    
    def _deflection_to_egocentric(self, deflection):
        # In order to calculate a drive, we need to express the funiculus deflection in the fly's (thorax) frame of reference. 
        # Since the pedicel is between the funiculus and the thorax in the kinematic chain, we can use it to convert the funiculus deflection to the fly's reference frame.
        pedicel_rotation_l = quat_inv(self.pedicel_angle_0[0])
        pedicel_rotation_r = quat_inv(self.pedicel_angle_0[1])

        egocentric_deflection_l = quat_rotate(pedicel_rotation_l, deflection[0])
        egocentric_deflection_r = quat_rotate(pedicel_rotation_r, deflection[1])

        return [egocentric_deflection_l, egocentric_deflection_r]
    
    def _generate_control_signal(self, egocentric_deflection, fwd_k=1, lat_k=1):
        # We use the deflection of the fly's antennae to generate a control signal that can be used to steer the fly.
        # We separate this drive into a magnitude and a lateral drive. The magnitude is proportional to the sum .
        lateral_drive = lat_k * (egocentric_deflection[0][1] - egocentric_deflection[1][1]) # assymetry: difference between the x components of each antenna's deflection
        forward_drive = fwd_k * (egocentric_deflection[0][0] + egocentric_deflection[1][0]) # magnitude: sum of the y components of each antenna's deflection
        return forward_drive, lateral_drive
    
    def process_wind(self, antenna_data):
        funiculus_position = self._funiculus_position(antenna_data)
        deflection = self._compute_deflection(funiculus_position)
        egocentric_deflection = self._deflection_to_egocentric(deflection)
        control_signal = self._generate_control_signal(egocentric_deflection)

        return control_signal