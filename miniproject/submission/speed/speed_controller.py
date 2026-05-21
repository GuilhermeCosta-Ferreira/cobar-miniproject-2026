# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import dump
from pathlib import Path
from itertools import product
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .utils import LEG_NAMES, run_simulation
from ..hybrid_controller import HybridTurningController

from miniproject.simulation import MiniprojectSimulation



# ================================================================
# 0. Section: FUNCTIONS
# ================================================================
def got_to_food(sim: MiniprojectSimulation) -> bool:
    banana_xy = sim.world.banana_xy
    fly_xy = np.array(sim.get_body_positions(sim.fly.name)[0][:2])
    dist = np.sqrt(np.sum((fly_xy - banana_xy) ** 2))
    return dist <= 3

def fell(sim: MiniprojectSimulation) -> bool:
    fly_body = sim.get_body_positions(sim.fly.name)[0][2]
    leg_pos = sim.get_body_positions(sim.fly.name)[-1][2]
    if fly_body < leg_pos:
        return True
    return False

def run_turning_simulation(
    action,
    seed=42,
    warmup_steps=500,
    n_cycles=12,
    frequency=12,
    timestep=1e-4,
):
    """Run a short turning simulation and return the pose history.

    Returns an array of shape (num_steps, 7) where columns are
    [x, y, z, qw, qx, qy, qz].
    """

    def step_callback(sim):
        fly_name = next(iter(sim.world.fly_lookup))
        body = sim.mj_data.body(f"{fly_name}/")
        return np.concatenate((body.xpos, body.xquat))

    num_steps = int(n_cycles / frequency / timestep)
    sim = MiniprojectSimulation(0, seed)
    controller = HybridTurningController(sim.timestep)
    joint_angles = np.zeros((num_steps, 42))
    adhesion = np.zeros((num_steps, 6))

    fell_count = 0
    for step in range(num_steps):
        if got_to_food(sim):
            print(f"Got to goal in {step} timesteps.")
            break

        if fell(sim):
            fell_count += 1
        if fell_count > 1000:
            print("Fly fell")
            break

        joint_angles[step], adhesion[step] = controller.step(sim, action)

    hist = run_simulation(
        dof_angles=joint_angles,
        adhesion_segments=[f"{leg}_tarsus5" for leg in LEG_NAMES],
        adhesion_signals=adhesion,
        step_callback=step_callback,
        warmup_steps=warmup_steps,
        skip_render=True,
        verbose=False,
    )[1]
    return np.array(hist)


def get_fwd_rot_vel(hists, timestep=1e-4):
    """Compute forward and rotational velocities from pose histories.

    Discards the first 2500 steps (warm-up transient), then estimates
    velocities from the remaining trajectory.

    Returns (vf, vr) — forward velocity (mm/s) and rotational velocity (rad/s).
    """
    hists = hists[..., 2500:, :]
    w, x, y, z = (hists[..., i] for i in range(3, 7))
    heading = 1 - 2 * (y * y + z * z) + 2 * (w * z + x * y) * 1j
    vf = (np.gradient(hists[..., :2] @ (1, 1j), axis=-1) / heading).real.mean(
        -1
    ) / timestep
    angles = np.unwrap(np.angle(heading), axis=-1)
    vr = (angles[..., -1] - angles[..., 0]) / ((angles.shape[-1] - 1) * timestep)
    return vf, vr



# ================================================================
# 0. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # Sweep over a grid of descending signals with multiple seeds
    min_drive = -2
    max_drive = 2
    n_drives = 9
    drives = np.linspace(min_drive, max_drive, n_drives)
    seeds = np.arange(3)

    hists = np.array(
        Parallel(n_jobs=-1)(
            delayed(run_turning_simulation)(np.array([l_drive, r_drive]), seed=seed)
            for l_drive, r_drive, seed in tqdm(list(product(drives, drives, seeds)))
        )
    )

    # Trajectory grid — each panel shows the 2D path for one (δ_L, δ_R) pair
    vmax = np.abs(hists).max()
    fig, ax = plt.subplots(figsize=(8, 8))
    for k, xy in enumerate(hists[..., :2]):
        i, j = np.unravel_index(k, (n_drives, n_drives, len(seeds)))[:2]
        p = np.array([i, j]) * vmax
        ax.scatter(*p, c="k", s=50, lw=0, marker=".")
        ax.plot(*(xy + p).T, color="k", alpha=0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()

    # Extract forward and rotational velocities, then plot heatmaps
    vf, vr = get_fwd_rot_vel(hists)
    im_vf = vf.reshape((n_drives, n_drives, len(seeds))).mean(axis=-1)
    im_vr = vr.reshape((n_drives, n_drives, len(seeds))).mean(axis=-1)
    half = (max_drive - min_drive) / (n_drives - 1) / 2
    extent = (min_drive - half, max_drive + half, min_drive - half, max_drive + half)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    vmax_f = np.abs(im_vf).max()
    im0 = axs[0].imshow(
        im_vf, origin="lower", extent=extent, vmax=vmax_f, vmin=-vmax_f, cmap="coolwarm"
    )
    vmax_r = np.abs(im_vr).max()
    im1 = axs[1].imshow(
        im_vr, origin="lower", extent=extent, vmax=vmax_r, vmin=-vmax_r, cmap="coolwarm"
    )

    for ax in axs:
        ax.set_xticks(drives)
        ax.set_yticks(drives)
        ax.set_ylabel("$\\delta_R$")
        ax.set_xlabel("$\\delta_L$")

    fig.colorbar(im0, ax=axs[0])
    fig.colorbar(im1, ax=axs[1])
    axs[0].set_title("Forward velocity (mm/s)")
    axs[1].set_title("Turning velocity (rad/s)")
    plt.show()

    model = make_pipeline(
        PolynomialFeatures(degree=3),
        LinearRegression(fit_intercept=False),
    )
    X = np.column_stack((vf, vr))
    Y = np.array(list(product(drives, drives, seeds)))[:, :2]
    model.fit(X, Y)

    r2 = r2_score(Y, model.predict(X), multioutput="raw_values")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    axs[0].scatter(Y[:, 0], model.predict(X)[:, 0], c="k", lw=0, marker=".")
    axs[0].set_xlabel("True $\\delta_L$")
    axs[0].set_ylabel("Predicted $\\delta_L$")
    axs[0].text(0.02, 0.98, f"$R^2$ = {r2[0]:.4f}", transform=axs[0].transAxes, va="top")
    axs[1].scatter(Y[:, 1], model.predict(X)[:, 1], c="k", lw=0, marker=".")
    axs[1].set_xlabel("True $\\delta_R$")
    axs[1].set_ylabel("Predicted $\\delta_R$")
    axs[1].text(0.02, 0.98, f"$R^2$ = {r2[1]:.4f}", transform=axs[1].transAxes, va="top")
    for ax in axs:
        ax.set_aspect(1)
    plt.show()

    # Test the inverse model: predict drives for target velocities, then simulate
    target_vf_vr = np.array([[8, 1], [10, 0], [5, 4]])
    predicted_drives = model.predict(target_vf_vr)

    actual_vf_vr = np.array(
        [
            [
                get_fwd_rot_vel(run_turning_simulation(lr_drive_pred, seed=seed))
                for seed in seeds
            ]
            for lr_drive_pred in predicted_drives
        ]
    )

    print("Target (v_f, v_θ) → Actual mean (v_f, v_θ):")
    for target, actual in zip(target_vf_vr, actual_vf_vr.mean(axis=-2)):
        print(f"  {target} → [{actual[0]:.2f}, {actual[1]:.2f}]")

    MODEL_PATH = Path("models/turning_inverse_model_flat.joblib")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    dump(model, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
