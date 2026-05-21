# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import tqdm
import json

import numpy as np

from pathlib import Path
from joblib import Parallel, delayed
from itertools import product
from copy import deepcopy

from miniproject.simulation import MiniprojectSimulation
from flygym.compose import ActuatorType

from ..controller import Controller
from ..periphery import SEEDS
from ..config import load_config



# ================================================================
# 1. Section: INPUTS
# ================================================================
MAX_NUM_STEPS: int = 100_000
LEVELS: list[int] = [2]

ROOT: Path = Path(__file__).resolve().parent
RESULTS_FOLDER: Path = ROOT / "optimization_results"

ROOT: Path = Path(__file__).resolve().parents[-1]
BASE_CONFIG_PATH: Path = ROOT / "config" / "config.yaml"

MIN_HEIGHT_LIST: list[float] = np.linspace(100, 400, 3).tolist()
SCARY_HEIGHT_LIST: list[float] = np.linspace(200, 500, 3).tolist()
VISION_GAIN_LIST: list[float] = np.linspace(2, 7, 3).tolist()
SLOW_DOWN_RATE_LIST: list[float] = np.linspace(0.3, 0.7, 3).tolist()
ALPHA_LIST: list[float] = np.linspace(0.01, 0.20, 3).tolist()



# ================================================================
# 2. Section: FUNCTIONS
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

def format_params(config: dict) -> str:
    vision = config["vision"]

    return (
        f"min_height={vision['min_height']}, "
        f"scary_height={vision['scary_height']}, "
        f"gain={vision['gain']}, "
        f"slow_down_rate={vision['slow_down_rate']}, "
        f"alpha={vision['alpha']}"
    )

def run_sim(
    level: int,
    seed: int,
    config: dict,
) -> dict:
    sim = MiniprojectSimulation(level, seed)
    controller = Controller(sim, config)

    fell_count = 0

    result = {
        "level": level,
        "seed": seed,
        "success": False,
        "steps": -1,
        "fell": False,
        "config": config,
    }

    params_str = format_params(controller.config)

    for step in range(MAX_NUM_STEPS):
        if got_to_food(sim):
            print(
                f"Got to goal in {step} timesteps. "
                f"Level={level}, seed={seed}, params: {params_str}"
            )
            result["success"] = True
            result["steps"] = step
            return result

        if fell(sim):
            fell_count += 1

        if fell_count > 1000:
            print(
                f"Fly fell at step {step}. "
                f"Level={level}, seed={seed}, params: {params_str}"
            )
            result["fell"] = True
            result["steps"] = step
            return result

        joint_angles, adhesion_signals = controller.step(sim)

        sim.set_actuator_inputs(
            sim.fly.name,
            ActuatorType.POSITION,
            joint_angles,
        )
        sim.set_actuator_inputs(
            sim.fly.name,
            ActuatorType.ADHESION,
            adhesion_signals,
        )

        sim.step()

    print(
        f"Took too long, reached {MAX_NUM_STEPS} steps. "
        f"Level={level}, seed={seed}, params: {params_str}"
    )
    result["steps"] = MAX_NUM_STEPS
    return result

def build_param_grid() -> list[dict]:
    configs = []
    base_config = load_config(BASE_CONFIG_PATH)

    for min_height, scary_height, gain, slow_down_rate, alpha in product(
        MIN_HEIGHT_LIST,
        SCARY_HEIGHT_LIST,
        VISION_GAIN_LIST,
        SLOW_DOWN_RATE_LIST,
        ALPHA_LIST,
    ):
        config = deepcopy(base_config)

        if min_height >= scary_height:
            continue

        config["vision"]["min_height"] = float(min_height)
        config["vision"]["scary_height"] = float(scary_height)
        config["vision"]["gain"] = float(gain)
        config["vision"]["slow_down_rate"] = float(slow_down_rate)
        config["vision"]["alpha"] = float(alpha)

        configs.append(config)

    return configs



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    SEEDS = SEEDS[1:4]

    param_grid = build_param_grid()

    print(f"Testing {len(param_grid)} parameter configs.")
    print(f"Testing {len(SEEDS)} seeds per config.")
    print(f"Total simulations: {len(param_grid) * len(SEEDS) * len(LEVELS)}")

    for level in LEVELS:
        jobs = []

        for config_id, config in enumerate(param_grid):
            for seed in SEEDS:
                jobs.append((level, seed, config_id, config))

        results = Parallel(n_jobs=-1)(
            delayed(run_sim)(
                level=level,
                seed=seed,
                config={
                    **config,
                    "config_id": config_id,
                },
            )
            for level, seed, config_id, config in tqdm.tqdm(jobs)
        )

        output_path = RESULTS_FOLDER / f"grid_search_level_{level}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Saved results to {output_path}")
