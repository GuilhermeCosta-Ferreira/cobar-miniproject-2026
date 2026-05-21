# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import tqdm
import json

import numpy as np

from pathlib import Path
from joblib import Parallel, delayed

from miniproject.simulation import MiniprojectSimulation
from submission.controller import Controller
from flygym.compose import ActuatorType
from submission.world import SEEDS



# ================================================================
# 1. Section: INPUTS
# ================================================================
MAX_NUM_STEPS: int = 100_000
LEVELS: list[int] = [1, 2, 3, 4]
RESULTS_FOLDER: Path = Path("eval_test")



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

def run_sim(level, seed, success_rate):
    sim = MiniprojectSimulation(level, seed)
    controller = Controller(sim)

    fell_count = 0
    for step in range(MAX_NUM_STEPS):
        if got_to_food(sim):
            print(f"Got to goal in {step} timesteps.")
            success_rate[seed] = step
            break

        if fell(sim):
            fell_count += 1

        if fell_count > 1000:
            print("Fly fell")
            success_rate[seed] = -1
            break

        joint_angles, adhesion_signals = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion_signals)
        sim.step()
    else:
        print("Took too long")
        success_rate[seed] = -1

    return success_rate



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for level in LEVELS:
        success_rate = {}

        success_rate = Parallel(n_jobs=-1)(
            delayed(run_sim)(level, seed, success_rate)
            for seed in tqdm.tqdm(SEEDS)
        )

        output_path = RESULTS_FOLDER / f"sucess_rates_level_{level}.json"
        with open(output_path, "w") as f:
            json.dump(success_rate, f, indent=4)

        print(f"Saved results to {output_path}")
