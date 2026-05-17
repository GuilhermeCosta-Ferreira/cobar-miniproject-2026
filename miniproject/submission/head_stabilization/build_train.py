# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import pickle

from tqdm import trange
from pathlib import Path
from typing import Optional
from dm_control.utils import transformations
from dm_control.rl.control import PhysicsError
from flygym.compose import ActuatorType

from miniproject.simulation import MiniprojectSimulation

from .simple_controller import SimpleController



# ================================================================
# 1. Section: Functions
# ================================================================
def run_simulation(
    dn_drive: np.ndarray = np.array([1, 1]),
    sim_duration: float = 0.5,
    output_dir: Optional[Path] = None,
):
    """Simulate locomotion and collect proprioceptive information to train
        a neural network for head stabilization.

    Parameters
    ----------
    dn_drive : np.ndarray, optional
        The DN drive values for the left and right wings. Defaults to
        [1, 1].
    sim_duration : float, optional
        The duration of the simulation in seconds. Defaults to 0.5.
    live_display : bool, optional
        If True, enables live display. Defaults to False.
    output_dir : Path, optional
        The directory to which output files are saved. Defaults to None.

    Raises
    ------
    ValueError
        Raised when an unknown terrain type is provided.
    """
    sim = MiniprojectSimulation(1, seed=42)
    controller = SimpleController(sim)

    sim.reset()
    info = {}
    info_hist, action_hist = [], []
    physics_error = False
    for _ in trange(int(sim_duration / sim.timestep)):
        action_hist.append(dn_drive)

        try:
            joint_angles, adhesion = controller.step(dn_drive)
            sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
            sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
            sim.step()
            sim.render_as_needed()
        except PhysicsError:
            print("Physics error detected!")
            physics_error = True
            break

        # TODO
        rendered_img = sim.renderer.frames

        # Get necessary angles
        info["roll"], info["pitch"], info["yaw"] = get_thorax_roll_pitch_yaw(sim)
        info_hist.append(info)

    # Save data if output_dir is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cam = sim.renderer
        cam.save_video(output_dir / "rendering.mp4")
        with open(output_dir / "sim_data.pkl", "wb") as f:
            data = {
                "info_hist": info_hist,
                "action_hist": action_hist,
                "errors": {
                    "physics_error": physics_error,
                },
            }
            pickle.dump(data, f)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_thorax_roll_pitch_yaw(sim: MiniprojectSimulation) -> tuple:
    # 1. Import the data
    fly_name = sim.fly.name
    body_segments = sim.fly.get_bodysegs_order()

    # 2. Find thorax index
    thorax_idx = next(
        i for i, seg in enumerate(body_segments)
        if seg.name == "c_thorax"
    )

    # 3. Get thorax quaternion from MuJoCo data
    quat = sim.get_body_rotations(fly_name)[thorax_idx].copy()

    # 4. Same convention as old code
    quat_inv = transformations.quat_inv(quat)
    roll, pitch, yaw = transformations.quat_to_euler(
        quat_inv,
        ordering="XYZ",
    )

    return roll, pitch, yaw
