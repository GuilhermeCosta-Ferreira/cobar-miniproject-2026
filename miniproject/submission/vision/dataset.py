# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import numpy as np

from pathlib import Path
from tqdm import tqdm
from flygym.compose import ActuatorType
from miniproject.simulation import MiniprojectSimulation

from ..world import SEEDS, SimpleController
from .visualize import produce_human_view

OUT_PATH = Path("submission/datasets")


# ================================================================
# 1. Section: Functions
# ================================================================
def generate_dataset(
    nr_iterations: int = int(1e5),
    out_path: Path = OUT_PATH,
    file_name: str = "dataset_level2",
    seeds: list[int] = SEEDS,
) -> Path:
    # 1. Run the dataset
    dataset = []
    for seed in tqdm(seeds):
        dataset = run_sim(dataset, seed, nr_iterations)
    dataset = np.asarray(dataset)

    # 2. Save the dataset
    os.makedirs(out_path, exist_ok=True)
    file_name = f"{file_name}_nr_seeds_{len(seeds)}_nr_ite_{nr_iterations}.npy"
    out_path = out_path / file_name
    np.save(out_path, dataset)

    return out_path


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def run_sim(
    dataset: list, seed: int, nr_iterations: int, drivers: list[float] = [1.0, 1.0]
) -> list:
    sim = MiniprojectSimulation(level=2, seed=seed)
    controller = SimpleController(sim)

    for i in range(nr_iterations):
        joint_angles, adhesion = controller.step(sim, np.array(drivers))
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()
        sim.render_as_needed()

        if i % 10000 == 0:
            img = produce_human_view(sim)
            dataset.append(img)

    return dataset


# ================================================================
# 2. Section: Main
# ================================================================
if __name__ == "__main__":
    generate_dataset(
        nr_iterations=100000,
    )
