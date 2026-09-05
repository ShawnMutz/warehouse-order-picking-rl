"""
Frozen Set-DQN step-limit sensitivity diagnostic at fifteen picks.

Evaluates the same frozen fourteen-pick curriculum policy on the same
50 fresh fifteen-pick uniform orders under MAX_STEPS=150 and 250.

No training occurs. The goal is to isolate whether the fixed 150-step
episode horizon is materially constraining completion at larger
cardinalities.

The script records:
- completion rate
- any-pick rate
- all-items-collected rate
- all-items-collected-but-not-completed rate
- mean items collected / collection fraction
- mean distance / completed distance
- mean steps
- invalid-action rate
- paired per-order outcome changes between 150 and 250 steps
"""

from utils.reward_shaping import get_remaining_pick_locations
from utils.order_generation import generate_order
from environment.warehouse_env import WarehouseEnv
from agents.set_dqn_agent import SetDQNAgent
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUN_NAME = "set_dqn_fifteen_step_limit_sensitivity"
EVALUATION_SEED = 42
ORDER_SIZE = 15
ORDERS = 50
DISTRIBUTION = "uniform"
PROBE_SEED_START = 340_000
MAX_STEPS_VALUES = [150, 250]
MAX_ORDER_SIZE = 20
DEVICE = "cpu"

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "set_dqn_curriculum_fourteen_pick_from_twelve_11_cycles"
    / "set_dqn_best_model.pt"
)

LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99999
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
TARGET_UPDATE = 1_000
HIDDEN_DIM = 256
PICK_EMBEDDING_DIM = 64


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_warehouse():
    height = 14
    width = 17

    grid = np.ones((height, width), dtype=np.int8)
    grid[0, :] = 0
    grid[13, :] = 0

    aisle_columns = [1, 3, 5, 7, 9, 11, 13, 15]

    for col in aisle_columns:
        grid[:, col] = 0

    depot = (0, 0)

    all_pick_locations = [
        (row, col)
        for col in aisle_columns
        for row in range(1, 13)
    ]

    if len(all_pick_locations) != 96:
        raise AssertionError("Expected exactly 96 pick locations.")

    return grid, depot, aisle_columns, all_pick_locations


def create_environment(grid, depot, all_pick_locations, max_steps):
    return WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=max_steps,
        move_cost=-1.0,
        invalid_penalty=-2.0,
        pick_reward=2.0,
        completion_reward=20.0,
    )


def create_agent(env):
    return SetDQNAgent(
        env=env,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        target_update=TARGET_UPDATE,
        hidden_dim=HIDDEN_DIM,
        pick_embedding_dim=PICK_EMBEDDING_DIM,
        max_order_size=MAX_ORDER_SIZE,
        device=DEVICE,
    )


def extract_policy_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format. Expected dictionary."
        )

    for key in [
        "policy_state_dict",
        "policy_net_state_dict",
        "policy_net",
        "model_state_dict",
        "state_dict",
    ]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    values = list(checkpoint.values())

    if values and all(torch.is_tensor(value) for value in values):
        return checkpoint

    raise RuntimeError(
        "Could not locate policy-network parameters.\n"
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def load_policy(agent, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Fourteen-pick checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print("\nLoading frozen fourteen-pick Set-DQN:")
    print(f"  {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    state_dict = extract_policy_state_dict(checkpoint)

    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)

    agent.policy_net.eval()
    agent.target_net.eval()

    print("Frozen fourteen-pick policy loaded successfully.")


def order_key(order):
    return tuple(sorted(tuple(location) for location in order))


def create_evaluation_orders(all_pick_locations, aisle_columns):
    orders = []
    seen_keys = set()
    seed = PROBE_SEED_START

    while len(orders) < ORDERS:
        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=ORDER_SIZE,
            distribution=DISTRIBUTION,
            seed=seed,
            aisle_columns=aisle_columns,
        )

        key = order_key(order)

        if key not in seen_keys:
            orders.append(
                {
                    "order_index": len(orders) + 1,
                    "seed": seed,
                    "order": list(key),
                }
            )
            seen_keys.add(key)

        seed += 1

    return orders


def get_collection_metrics(
    environment_state,
    all_pick_locations,
):
    remaining_picks = get_remaining_pick_locations(
        environment_state,
        all_pick_locations,
    )

    items_remaining = len(remaining_picks)
    items_collected = ORDER_SIZE - items_remaining

    items_collected = max(
        0,
        min(ORDER_SIZE, items_collected),
    )

    items_remaining = max(
        0,
        min(ORDER_SIZE, items_remaining),
    )

    collection_fraction = items_collected / ORDER_SIZE

    return (
        items_collected,
        items_remaining,
        float(collection_fraction),
    )


def calculate_action_metrics(steps, travel_distance):
    valid_actions = int(round(travel_distance))
    invalid_actions = max(0, int(steps) - valid_actions)

    if steps > 0:
        invalid_action_rate = (
            100.0 * invalid_actions / steps
        )
    else:
        invalid_action_rate = 0.0

    return (
        valid_actions,
        invalid_actions,
        float(invalid_action_rate),
    )


def evaluate_order(
    env,
    agent,
    order,
    all_pick_locations,
):
    state = env.reset(order)

    if state.shape != (98,):
        raise RuntimeError(
            f"Expected raw state shape (98,), received {state.shape}."
        )

    terminated = False
    truncated = False

    start_time = time.perf_counter()

    while not (terminated or truncated):
        action = agent.select_action(
            state,
            eval_mode=True,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        state = next_state

    runtime = time.perf_counter() - start_time

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        state,
        all_pick_locations,
    )

    (
        valid_actions,
        invalid_actions,
        invalid_action_rate,
    ) = calculate_action_metrics(
        env.steps,
        env.travel_distance,
    )

    all_items_collected = (
        items_collected == ORDER_SIZE
    )

    no_finish = (
        all_items_collected
        and not bool(terminated)
    )

    return {
        "completed": bool(terminated),
        "truncated": bool(truncated),
        "all_items_collected": bool(all_items_collected),
        "collected_all_but_not_completed": bool(no_finish),
        "items_collected": items_collected,
        "items_remaining": items_remaining,
        "collection_fraction": collection_fraction,
        "distance": float(env.travel_distance),
        "steps": int(env.steps),
        "valid_actions": valid_actions,
        "invalid_actions": invalid_actions,
        "invalid_action_rate": invalid_action_rate,
        "runtime": float(runtime),
    }


def evaluate_horizon(
    env,
    agent,
    evaluation_orders,
    all_pick_locations,
    max_steps,
):
    raw_results = []

    print(
        "\n"
        "------------------------------------------------\n"
        f"EVALUATING FIFTEEN PICKS WITH MAX_STEPS={max_steps}\n"
        "------------------------------------------------"
    )

    for row in evaluation_orders:
        result = evaluate_order(
            env,
            agent,
            row["order"],
            all_pick_locations,
        )

        raw_results.append(
            {
                "max_steps": max_steps,
                "order_size": ORDER_SIZE,
                "order_index": row["order_index"],
                "seed": row["seed"],
                "distribution": DISTRIBUTION,
                "order": str(row["order"]),
                **result,
            }
        )

        index = row["order_index"]

        if (
            index == 1
            or index % 10 == 0
            or index == len(evaluation_orders)
        ):
            print(
                f"Order {index:02d}/{len(evaluation_orders)}"
                f" | complete {int(result['completed'])}"
                f" | collect {result['items_collected']}/{ORDER_SIZE}"
                f" | distance {result['distance']:.0f}"
                f" | steps {result['steps']}"
                f" | invalid {result['invalid_action_rate']:.1f}%"
            )

    completed_count = sum(
        row["completed"]
        for row in raw_results
    )

    completion_rate = (
        100.0 * completed_count / len(raw_results)
    )

    any_pick_count = sum(
        row["items_collected"] > 0
        for row in raw_results
    )

    any_pick_rate = (
        100.0 * any_pick_count / len(raw_results)
    )

    all_items_collected_count = sum(
        row["all_items_collected"]
        for row in raw_results
    )

    all_items_collected_rate = (
        100.0
        * all_items_collected_count
        / len(raw_results)
    )

    no_finish_count = sum(
        row["collected_all_but_not_completed"]
        for row in raw_results
    )

    no_finish_rate = (
        100.0
        * no_finish_count
        / len(raw_results)
    )

    mean_items_collected = float(
        np.mean(
            [
                row["items_collected"]
                for row in raw_results
            ]
        )
    )

    mean_collection_fraction = float(
        np.mean(
            [
                row["collection_fraction"]
                for row in raw_results
            ]
        )
    )

    mean_distance = float(
        np.mean(
            [
                row["distance"]
                for row in raw_results
            ]
        )
    )

    completed_distances = [
        row["distance"]
        for row in raw_results
        if row["completed"]
    ]

    mean_completed_distance = (
        float(np.mean(completed_distances))
        if completed_distances
        else np.nan
    )

    mean_steps = float(
        np.mean(
            [
                row["steps"]
                for row in raw_results
            ]
        )
    )

    mean_invalid_action_rate = float(
        np.mean(
            [
                row["invalid_action_rate"]
                for row in raw_results
            ]
        )
    )

    summary = {
        "max_steps": max_steps,
        "order_size": ORDER_SIZE,
        "distribution": DISTRIBUTION,
        "orders": len(raw_results),
        "completed_orders": completed_count,
        "completion_rate": completion_rate,
        "truncated_orders": sum(
            row["truncated"]
            for row in raw_results
        ),
        "any_pick_count": any_pick_count,
        "any_pick_rate": any_pick_rate,
        "all_items_collected_count":
            all_items_collected_count,
        "all_items_collected_rate":
            all_items_collected_rate,
        "collected_all_but_not_completed_count":
            no_finish_count,
        "collected_all_but_not_completed_rate":
            no_finish_rate,
        "mean_items_collected":
            mean_items_collected,
        "mean_collection_fraction":
            mean_collection_fraction,
        "mean_distance":
            mean_distance,
        "mean_completed_distance":
            mean_completed_distance,
        "mean_steps":
            mean_steps,
        "mean_invalid_action_rate":
            mean_invalid_action_rate,
    }

    return summary, raw_results


def build_paired_comparison(raw_by_horizon):
    low_steps = min(MAX_STEPS_VALUES)
    high_steps = max(MAX_STEPS_VALUES)

    low_rows = {
        row["order_index"]: row
        for row in raw_by_horizon[low_steps]
    }

    high_rows = {
        row["order_index"]: row
        for row in raw_by_horizon[high_steps]
    }

    paired_rows = []

    for order_index in range(1, ORDERS + 1):
        low = low_rows[order_index]
        high = high_rows[order_index]

        paired_rows.append(
            {
                "order_index": order_index,
                "seed": low["seed"],
                "order": low["order"],
                "completed_150": low["completed"],
                "completed_250": high["completed"],
                "all_items_collected_150":
                    low["all_items_collected"],
                "all_items_collected_250":
                    high["all_items_collected"],
                "no_finish_150":
                    low["collected_all_but_not_completed"],
                "no_finish_250":
                    high["collected_all_but_not_completed"],
                "items_collected_150":
                    low["items_collected"],
                "items_collected_250":
                    high["items_collected"],
                "distance_150":
                    low["distance"],
                "distance_250":
                    high["distance"],
                "steps_150":
                    low["steps"],
                "steps_250":
                    high["steps"],
                "invalid_rate_150":
                    low["invalid_action_rate"],
                "invalid_rate_250":
                    high["invalid_action_rate"],
                "failed_150_completed_250":
                    (
                        not low["completed"]
                        and high["completed"]
                ),
                "no_finish_150_completed_250":
                    (
                        low[
                            "collected_all_but_not_completed"
                        ]
                        and high["completed"]
                ),
                "completion_outcome_changed":
                    (
                        low["completed"]
                        != high["completed"]
                ),
            }
        )

    return paired_rows


def save_csv(rows, output_path):
    if not rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    print(
        "\n"
        "================================================\n"
        "SET-DQN FIFTEEN-PICK STEP-LIMIT SENSITIVITY\n"
        "================================================"
    )

    set_random_seeds(EVALUATION_SEED)

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    ) = create_warehouse()

    base_env = create_environment(
        grid,
        depot,
        all_pick_locations,
        min(MAX_STEPS_VALUES),
    )

    agent = create_agent(base_env)

    load_policy(
        agent,
        SOURCE_MODEL_PATH,
    )

    test_state = base_env.reset(
        all_pick_locations[:ORDER_SIZE]
    )

    test_q_values = agent.get_q_values(
        test_state
    )

    if test_state.shape != (98,):
        raise RuntimeError(
            f"Expected state shape (98,), "
            f"received {test_state.shape}."
        )

    if test_q_values.shape != (4,):
        raise RuntimeError(
            f"Expected four Q-values, "
            f"received {test_q_values.shape}."
        )

    print(
        f"\nRun name:               {RUN_NAME}\n"
        f"Model:                  fourteen-pick curriculum best\n"
        f"Policy frozen:          yes\n"
        f"Training during eval:   no\n"
        f"Distribution:           {DISTRIBUTION}\n"
        f"Order size:             {ORDER_SIZE}\n"
        f"Orders:                 {ORDERS}\n"
        f"Probe seed start:       {PROBE_SEED_START}\n"
        f"Max-steps values:       {MAX_STEPS_VALUES}\n"
        f"Raw state dimension:    {test_state.shape[0]}\n"
        f"Max model order size:   {MAX_ORDER_SIZE}\n"
        f"Device:                 {DEVICE}\n"
    )

    print(
        "Creating fresh paired fifteen-pick evaluation orders..."
    )

    evaluation_orders = create_evaluation_orders(
        all_pick_locations,
        aisle_columns,
    )

    print(
        f"Created {len(evaluation_orders)} unique "
        f"{ORDER_SIZE}-pick {DISTRIBUTION} orders."
    )

    results_directory = (
        PROJECT_ROOT
        / "results"
        / "development"
        / RUN_NAME
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        results_directory
        / "experiment_config.json"
    )

    orders_path = (
        results_directory
        / "evaluation_orders.csv"
    )

    summary_path = (
        results_directory
        / "step_limit_summary.csv"
    )

    raw_path = (
        results_directory
        / "step_limit_raw.csv"
    )

    paired_path = (
        results_directory
        / "paired_step_limit_comparison.csv"
    )

    config = {
        "run_name": RUN_NAME,
        "evaluation_type":
            "frozen_fifteen_pick_step_limit_sensitivity",
        "purpose":
            "isolate_effect_of_episode_horizon_on_large_order_completion",
        "source_model":
            str(SOURCE_MODEL_PATH),
        "source_training_order_size":
            14,
        "policy_frozen":
            True,
        "additional_training":
            False,
        "order_size":
            ORDER_SIZE,
        "orders":
            ORDERS,
        "distribution":
            DISTRIBUTION,
        "probe_seed_start":
            PROBE_SEED_START,
        "max_steps_values":
            MAX_STEPS_VALUES,
        "evaluation_seed":
            EVALUATION_SEED,
        "max_order_size":
            MAX_ORDER_SIZE,
        "paired_orders":
            True,
        "state_includes_step_counter":
            False,
        "exact_benchmark":
            False,
        "hidden_dim":
            HIDDEN_DIM,
        "pick_embedding_dim":
            PICK_EMBEDDING_DIM,
        "device":
            DEVICE,
    }

    with config_path.open("w") as file:
        json.dump(
            config,
            file,
            indent=4,
        )

    order_rows = [
        {
            "order_index":
                row["order_index"],
            "seed":
                row["seed"],
            "order_size":
                ORDER_SIZE,
            "distribution":
                DISTRIBUTION,
            "order":
                str(row["order"]),
        }
        for row in evaluation_orders
    ]

    save_csv(
        order_rows,
        orders_path,
    )

    summaries = []
    all_raw_results = []
    raw_by_horizon = {}

    total_start_time = (
        time.perf_counter()
    )

    for max_steps in MAX_STEPS_VALUES:
        env = create_environment(
            grid,
            depot,
            all_pick_locations,
            max_steps,
        )

        (
            summary,
            raw_results,
        ) = evaluate_horizon(
            env,
            agent,
            evaluation_orders,
            all_pick_locations,
            max_steps,
        )

        summaries.append(summary)
        raw_by_horizon[max_steps] = raw_results
        all_raw_results.extend(raw_results)

        print(
            "\n"
            f"MAX_STEPS={max_steps} SUMMARY\n"
            "------------------------------------------------"
        )

        print(
            f"Completion:              "
            f"{summary['completion_rate']:.1f}% "
            f"({summary['completed_orders']}/"
            f"{summary['orders']})"
        )

        print(
            f"Any-pick rate:           "
            f"{summary['any_pick_rate']:.1f}%"
        )

        print(
            f"All-items collected:     "
            f"{summary['all_items_collected_rate']:.1f}%"
        )

        print(
            f"All picked, no finish:   "
            f"{summary['collected_all_but_not_completed_rate']:.1f}%"
        )

        print(
            f"Mean items:              "
            f"{summary['mean_items_collected']:.3f}"
            f"/{ORDER_SIZE}"
        )

        print(
            f"Collection fraction:     "
            f"{summary['mean_collection_fraction']:.3f}"
        )

        print(
            f"Mean distance:           "
            f"{summary['mean_distance']:.2f}"
        )

        print(
            f"Mean completed distance: "
            f"{summary['mean_completed_distance']:.2f}"
        )

        print(
            f"Mean steps:              "
            f"{summary['mean_steps']:.2f}"
        )

        print(
            f"Invalid-action rate:     "
            f"{summary['mean_invalid_action_rate']:.1f}%"
        )

        save_csv(
            summaries,
            summary_path,
        )

        save_csv(
            all_raw_results,
            raw_path,
        )

    paired_rows = build_paired_comparison(
        raw_by_horizon
    )

    save_csv(
        paired_rows,
        paired_path,
    )

    low_steps = min(MAX_STEPS_VALUES)
    high_steps = max(MAX_STEPS_VALUES)

    low_summary = next(
        row
        for row in summaries
        if row["max_steps"] == low_steps
    )

    high_summary = next(
        row
        for row in summaries
        if row["max_steps"] == high_steps
    )

    completion_gain_pp = (
        high_summary["completion_rate"]
        - low_summary["completion_rate"]
    )

    rescued_failures = sum(
        row["failed_150_completed_250"]
        for row in paired_rows
    )

    rescued_no_finish = sum(
        row["no_finish_150_completed_250"]
        for row in paired_rows
    )

    changed_outcomes = sum(
        row["completion_outcome_changed"]
        for row in paired_rows
    )

    total_runtime = (
        time.perf_counter()
        - total_start_time
    )

    print(
        "\n"
        "================================================\n"
        "FIFTEEN-PICK STEP-LIMIT SENSITIVITY COMPLETE\n"
        "================================================"
    )

    print(
        f"Total evaluation runtime: "
        f"{total_runtime:.2f} seconds\n"
    )

    print(
        "Paired horizon summary:"
    )

    for summary in summaries:
        print(
            f"  MAX_STEPS={summary['max_steps']:3d}"
            f" | complete "
            f"{summary['completion_rate']:5.1f}%"
            f" | collection "
            f"{summary['mean_collection_fraction']:.3f}"
            f" | all-picked "
            f"{summary['all_items_collected_rate']:5.1f}%"
            f" | no-finish "
            f"{summary['collected_all_but_not_completed_rate']:5.1f}%"
            f" | invalid "
            f"{summary['mean_invalid_action_rate']:5.1f}%"
        )

    print(
        "\nPaired effect of extending the horizon:"
    )

    print(
        f"  Completion gain: "
        f"{completion_gain_pp:+.1f} percentage points"
    )

    print(
        f"  Failed at {low_steps} but completed at {high_steps}: "
        f"{rescued_failures}/{ORDERS}"
    )

    print(
        f"  All-picked/no-finish at {low_steps}, "
        f"then completed at {high_steps}: "
        f"{rescued_no_finish}/{ORDERS}"
    )

    print(
        f"  Orders with changed completion outcome: "
        f"{changed_outcomes}/{ORDERS}"
    )

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Config:\n"
        f"    {config_path}"
    )

    print(
        f"  Evaluation orders:\n"
        f"    {orders_path}"
    )

    print(
        f"  Step-limit summary:\n"
        f"    {summary_path}"
    )

    print(
        f"  Raw results:\n"
        f"    {raw_path}"
    )

    print(
        f"  Paired comparison:\n"
        f"    {paired_path}"
    )


if __name__ == "__main__":
    main()
