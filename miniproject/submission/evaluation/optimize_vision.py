# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import json
import optuna

import numpy as np

from pathlib import Path

from miniproject.simulation import MiniprojectSimulation
from flygym.compose import ActuatorType

from ..controller import Controller
from ..config import load_config

# ================================================================
# 1. Section: INPUTS
# ================================================================
MAX_NUM_STEPS: int = 100_000
LEVEL: int = 2
SEED: int = 1
N_TRIAL_WORKERS: int = 1

ROOT: Path = Path(__file__).resolve().parent
RESULTS_FOLDER: Path = ROOT / "optimization_results"

ROOT: Path = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH: Path = ROOT / "config" / "stable_config.yaml"

MIN_HEIGHT_LIST: list[float] = np.linspace(100, 400, 3).tolist()
SCARE_HEIGHT_LIST: list[float] = np.linspace(200, 500, 3).tolist()
VISION_GAIN_LIST: list[float] = np.linspace(2, 7, 3).tolist()
SLOW_DOWN_RATE_LIST: list[float] = np.linspace(0.3, 0.7, 3).tolist()
ALPHA_LIST: list[float] = np.linspace(0.01, 0.20, 3).tolist()


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def score_result(result: dict) -> float:
    if result["fell"]:
        return -1000.0

    if not result["success"]:
        return -500.0

    # Faster success is better
    return 1000.0 - 0.01 * result["steps"]


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

    for step in range(MAX_NUM_STEPS):
        if got_to_food(sim):
            print(f"Got to goal in {step} timesteps. ")
            result["success"] = True
            result["steps"] = step
            return result

        if fell(sim):
            fell_count += 1

        if fell_count > 1000:
            print(f"Fly fell at step {step}. ")
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

    print(f"Took too long, reached {MAX_NUM_STEPS} steps. ")
    result["steps"] = MAX_NUM_STEPS
    return result


def build_config_from_trial(trial: optuna.Trial) -> dict:
    config = load_config(BASE_CONFIG_PATH)

    # Keep known-good stable values from your previous Level 1 search
    config["controller"]["dropoff_vt"] = trial.suggest_float("dropoff_vt", 1.0, 5.0)

    # Optional: correction-vector parameters
    config["vision"] = {
        "min_height": trial.suggest_float("min_height", 100, 400),
        "scare_height": trial.suggest_float("scare_height", 200, 500),
        "gain": trial.suggest_float("gain", 0.5, 5),
        "slow_down_rate": trial.suggest_float("slow_down_rate", 0, 1),
        "alpha": trial.suggest_float("alpha", 0, 0.5),
    }

    if config["vision"]["scare_height"] < config["vision"]["min_height"]:
        raise optuna.TrialPruned()

    return config


def objective(trial: optuna.Trial) -> float:
    config = build_config_from_trial(trial)

    scores = []
    results = []

    result = run_sim(
        level=LEVEL,
        seed=SEED,
        config=config,
    )

    result_score = score_result(result)

    scores.append(result_score)
    results.append(result)

    # Report intermediate mean score
    trial.report(float(np.mean(scores)), step=len(scores))

    # Stop bad trials early
    if trial.should_prune():
        raise optuna.TrialPruned()

    mean_score = float(np.mean(scores))

    trial.set_user_attr("results", results)
    trial.set_user_attr("config", config)

    return mean_score


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        study_name="vision_optimization",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=1,
        ),
    )

    study.optimize(
        objective,
        n_trials=100,
        n_jobs=N_TRIAL_WORKERS,
    )

    print("Best score:", study.best_value)
    print("Best params:")
    print(study.best_params)

    output_path = RESULTS_FOLDER / "optuna_vision_best.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "best_score": study.best_value,
                "best_params": study.best_params,
                "best_config": study.best_trial.user_attrs["config"],
                "best_results": study.best_trial.user_attrs["results"],
            },
            f,
            indent=4,
        )

    print(f"Saved best result to {output_path}")
