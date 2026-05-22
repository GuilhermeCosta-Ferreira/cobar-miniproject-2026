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
LEVELS: list[int] = [1]
SEED: int = 1
N_TRIAL_WORKERS: int = min(8, os.cpu_count())

ROOT: Path = Path(__file__).resolve().parent
RESULTS_FOLDER: Path = ROOT / "optimization_results"

ROOT: Path = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH: Path = ROOT / "config" / "config.yaml"

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


def format_params(config: dict) -> str:
    vision = config["vision"]

    return (
        f"min_height={vision['min_height']}, "
        f"scare_height={vision['scare_height']}, "
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


def build_config_from_trial(trial: optuna.Trial) -> dict:
    config = load_config(BASE_CONFIG_PATH)

    # Keep known-good stable values from your previous Level 1 search
    config["controller"]["dropoff_vt"] = trial.suggest_float("dropoff_vt", 1.0, 5.0)

    # Optional: correction-vector parameters
    correction_range = 0.03

    config["hybrid"] = {
        "f": {
            "coxa_lift": 0.0,
            "coxa_roll": 0.0,
            "coxa_yaw": trial.suggest_float(
                "f_coxa_yaw", -correction_range, correction_range
            ),
            "femur_lift": trial.suggest_float(
                "f_femur_lift", -correction_range, correction_range
            ),
            "femur_roll": 0.0,
            "tibia_lift": trial.suggest_float(
                "f_tibia_lift", -correction_range, correction_range
            ),
            "tarsus1_lift": trial.suggest_float(
                "f_tarsus1_lift", -correction_range, correction_range
            ),
        },
        "m": {
            "coxa_lift": 0.0,
            "coxa_roll": 0.0,
            "coxa_yaw": trial.suggest_float(
                "m_coxa_yaw", -correction_range, correction_range
            ),
            "femur_lift": trial.suggest_float(
                "m_femur_lift", -correction_range, correction_range
            ),
            "femur_roll": 0.0,
            "tibia_lift": trial.suggest_float(
                "m_tibia_lift", -correction_range, correction_range
            ),
            "tarsus1_lift": trial.suggest_float(
                "m_tarsus1_lift", -correction_range, correction_range
            ),
        },
        "h": {
            "coxa_lift": 0.0,
            "coxa_roll": 0.0,
            "coxa_yaw": trial.suggest_float(
                "h_coxa_yaw", -correction_range, correction_range
            ),
            "femur_lift": trial.suggest_float(
                "h_femur_lift", -correction_range, correction_range
            ),
            "femur_roll": 0.0,
            "tibia_lift": trial.suggest_float(
                "h_tibia_lift", -correction_range, correction_range
            ),
            "tarsus1_lift": trial.suggest_float(
                "h_tarsus1_lift", -correction_range, correction_range
            ),
        },
    }

    """
    config["hybrid"] = {
        "f": {
            "coxa_lift": trial.suggest_float("f_coxa_lift", -correction_range, correction_range),
            "coxa_roll": trial.suggest_float("f_coxa_roll", -correction_range, correction_range),
            "coxa_yaw": trial.suggest_float("f_coxa_yaw", -correction_range, correction_range),
            "femur_lift": trial.suggest_float("f_femur_lift", -correction_range, correction_range),
            "femur_roll": trial.suggest_float("f_femur_roll", -correction_range, correction_range),
            "tibia_lift": trial.suggest_float("f_tibia_lift", -correction_range, correction_range),
            "tarsus1_lift": trial.suggest_float("f_tarsus1_lift", -correction_range, correction_range),
        },

        "m": {
            "coxa_lift": trial.suggest_float("m_coxa_lift", -correction_range, correction_range),
            "coxa_roll": trial.suggest_float("m_coxa_roll", -correction_range, correction_range),
            "coxa_yaw": trial.suggest_float("m_coxa_yaw", -correction_range, correction_range),
            "femur_lift": trial.suggest_float("m_femur_lift", -correction_range, correction_range),
            "femur_roll": trial.suggest_float("m_femur_roll", -correction_range, correction_range),
            "tibia_lift": trial.suggest_float("m_tibia_lift", -correction_range, correction_range),
            "tarsus1_lift": trial.suggest_float("m_tarsus1_lift", -correction_range, correction_range),
        },

        "h": {
            "coxa_lift": trial.suggest_float("h_coxa_lift", -correction_range, correction_range),
            "coxa_roll": trial.suggest_float("h_coxa_roll", -correction_range, correction_range),
            "coxa_yaw": trial.suggest_float("h_coxa_yaw", -correction_range, correction_range),
            "femur_lift": trial.suggest_float("h_femur_lift", -correction_range, correction_range),
            "femur_roll": trial.suggest_float("h_femur_roll", -correction_range, correction_range),
            "tibia_lift": trial.suggest_float("h_tibia_lift", -correction_range, correction_range),
            "tarsus1_lift": trial.suggest_float("h_tarsus1_lift", -correction_range, correction_range),
        },
    }
    """

    return config


def objective(trial: optuna.Trial) -> float:
    config = build_config_from_trial(trial)

    scores = []
    results = []

    result = run_sim(
        level=LEVELS[0],
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
        study_name="level_1_controller_optimization",
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

    output_path = RESULTS_FOLDER / "optuna_level_1_best.json"

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
