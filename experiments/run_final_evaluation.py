"""
Final evaluation for the warehouse order-picking project.

Frozen model:
    results/development/
    set_dqn_mixed_consolidation_5_10_15_20_3_cycles/
    set_dqn_best_model.pt

Conditions:
    order sizes 5, 10, 15, 20
    distributions uniform and clustered
    50 fresh orders per condition
    400 orders total

Every order is evaluated by:
    DQN, S-shape, Return, Largest Gap, Exact

For incomplete DQN episodes, partial distance is stored only as a
diagnostic. DQN route distance, optimality gap, and DQN-vs-heuristic
comparisons are only calculated for successfully completed routes.
"""

from utils.reward_shaping import get_remaining_pick_locations
from utils.order_generation import generate_order
from exact.ratliff_rosenthal import exact_optimal_distance
from environment.warehouse_env import WarehouseEnv
from agents.set_dqn_agent import SetDQNAgent
import csv
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# HEURISTIC IMPORTS
# ============================================================

def resolve_callable(name, candidates):
    errors = []
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        fn = getattr(module, function_name, None)
        if callable(fn):
            print(f"Resolved {name}: {module_name}.{function_name}")
            return fn

        errors.append(
            f"{module_name}.{function_name}: callable not found"
        )

    raise ImportError(
        f"Could not resolve {name} heuristic.\nTried:\n  "
        + "\n  ".join(errors)
    )


S_SHAPE_ROUTE = resolve_callable(
    "S-shape",
    [
        ("heuristics.s_shape", "s_shape_route"),
        ("heuristics.s_shape", "s_shape_distance"),
        ("heuristics.s_shape", "s_shape"),
        ("heuristics.s_shape_route", "s_shape_route"),
        ("heuristics.s_shape_heuristic", "s_shape_route"),
        ("heuristics.s_shape_heuristic", "s_shape_heuristic"),
    ],
)

RETURN_ROUTE = resolve_callable(
    "Return",
    [
        ("heuristics.return_route", "return_route"),
        ("heuristics.return_route", "return_distance"),
        ("heuristics.return_heuristic", "return_route"),
        ("heuristics.return_heuristic", "return_heuristic"),
        ("heuristics.return_strategy", "return_route"),
        ("heuristics.return_strategy", "return_strategy"),
        ("heuristics.return_routing", "return_route"),
        ("heuristics.return_routing", "return_distance"),
        ("heuristics.return", "return_route"),
    ],
)

LARGEST_GAP_ROUTE = resolve_callable(
    "Largest Gap",
    [
        ("heuristics.largest_gap", "largest_gap_route"),
        ("heuristics.largest_gap", "largest_gap_distance"),
        ("heuristics.largest_gap", "largest_gap"),
        ("heuristics.largest_gap_route", "largest_gap_route"),
        ("heuristics.largest_gap_heuristic", "largest_gap_route"),
        ("heuristics.largest_gap_heuristic", "largest_gap_heuristic"),
    ],
)


# ============================================================
# SETTINGS
# ============================================================

RUN_NAME = "final_evaluation"

ORDER_SIZES = [5, 10, 15, 20]
DISTRIBUTIONS = ["uniform", "clustered"]
ORDERS_PER_CONDITION = 50

MAX_ORDER_SIZE = 20
MAX_STEPS = 250
EVALUATION_SEED = 42
DEVICE = "cpu"

FINAL_SEED_STARTS = {
    ("uniform", 5): 500_000,
    ("uniform", 10): 510_000,
    ("uniform", 15): 520_000,
    ("uniform", 20): 530_000,
    ("clustered", 5): 540_000,
    ("clustered", 10): 550_000,
    ("clustered", 15): 560_000,
    ("clustered", 20): 570_000,
}

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "set_dqn_mixed_consolidation_5_10_15_20_3_cycles"
    / "set_dqn_best_model.pt"
)

LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON = 0.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99999
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
TARGET_UPDATE = 1_000
HIDDEN_DIM = 256
PICK_EMBEDDING_DIM = 64

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "final" / RUN_NAME
)
CONFIG_PATH = RESULTS_DIR / "experiment_config.json"
ORDERS_PATH = RESULTS_DIR / "final_orders.csv"
RAW_PATH = RESULTS_DIR / "final_results.csv"
SUMMARY_PATH = RESULTS_DIR / "condition_summary.csv"
LONG_PATH = RESULTS_DIR / "method_long.csv"
PAIRED_PATH = RESULTS_DIR / "paired_dqn_comparisons.csv"


# ============================================================
# WAREHOUSE / MODEL
# ============================================================

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_warehouse():
    grid = np.ones((14, 17), dtype=np.int8)
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
        raise AssertionError("Expected 96 pick locations.")

    return grid, depot, aisle_columns, all_pick_locations


def create_environment(grid, depot, all_pick_locations):
    return WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=MAX_STEPS,
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
        epsilon=EPSILON,
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
        raise RuntimeError("Unexpected checkpoint format.")

    for key in [
        "policy_state_dict",
        "policy_net_state_dict",
        "policy_net",
        "model_state_dict",
        "state_dict",
    ]:
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate

    values = list(checkpoint.values())
    if values and all(torch.is_tensor(x) for x in values):
        return checkpoint

    raise RuntimeError(
        "Could not locate policy-network state dict. "
        f"Keys: {list(checkpoint.keys())}"
    )


def load_final_policy(agent):
    if not SOURCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final DQN checkpoint not found:\n{SOURCE_MODEL_PATH}"
        )

    print("\nLoading frozen final Set-DQN:")
    print(f"  {SOURCE_MODEL_PATH}")

    try:
        checkpoint = torch.load(
            SOURCE_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            SOURCE_MODEL_PATH,
            map_location=DEVICE,
        )

    state_dict = extract_policy_state_dict(checkpoint)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    agent.target_net.eval()
    agent.epsilon = 0.0

    print("Frozen final Set-DQN loaded successfully.")


# ============================================================
# ORDER GENERATION
# ============================================================

def order_key(order):
    return tuple(sorted(tuple(x) for x in order))


def generate_condition_orders(
    all_pick_locations,
    aisle_columns,
    distribution,
    order_size,
):
    seed = FINAL_SEED_STARTS[(distribution, order_size)]
    seen = set()
    rows = []

    while len(rows) < ORDERS_PER_CONDITION:
        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=order_size,
            distribution=distribution,
            seed=seed,
            aisle_columns=aisle_columns,
        )

        key = order_key(order)

        if key not in seen:
            rows.append(
                {
                    "distribution": distribution,
                    "order_size": order_size,
                    "condition_index": len(rows) + 1,
                    "seed": seed,
                    "order": list(key),
                }
            )
            seen.add(key)

        seed += 1

    return rows


def create_all_final_orders(
    all_pick_locations,
    aisle_columns,
):
    rows = []

    for distribution in DISTRIBUTIONS:
        for order_size in ORDER_SIZES:
            rows.extend(
                generate_condition_orders(
                    all_pick_locations,
                    aisle_columns,
                    distribution,
                    order_size,
                )
            )

    expected = (
        len(DISTRIBUTIONS)
        * len(ORDER_SIZES)
        * ORDERS_PER_CONDITION
    )

    if len(rows) != expected:
        raise AssertionError(
            f"Expected {expected} final orders, got {len(rows)}."
        )

    return rows


# ============================================================
# METRICS
# ============================================================

def collection_metrics(state, all_pick_locations, order_size):
    remaining = get_remaining_pick_locations(
        state,
        all_pick_locations,
    )

    items_remaining = len(remaining)
    items_collected = order_size - items_remaining
    items_collected = max(0, min(order_size, items_collected))

    return (
        items_collected,
        order_size - items_collected,
        float(items_collected / order_size),
    )


def action_metrics(steps, travel_distance):
    valid = int(round(travel_distance))
    invalid = max(0, int(steps) - valid)

    rate = (
        100.0 * invalid / steps
        if steps > 0
        else 0.0
    )

    return valid, invalid, float(rate)


def gap(distance, optimum):
    return (
        (distance - optimum)
        / optimum
        * 100.0
    )


def improvement(dqn_distance, heuristic_distance):
    """
    Positive = DQN shorter.
    Negative = DQN longer.
    """
    return (
        (heuristic_distance - dqn_distance)
        / heuristic_distance
        * 100.0
    )


def safe_mean(values):
    vals = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]
    return float(np.mean(vals)) if vals else np.nan


def safe_median(values):
    vals = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]
    return float(np.median(vals)) if vals else np.nan


# ============================================================
# HEURISTIC EXECUTION
# ============================================================

def coerce_distance(value, method_name):
    if isinstance(value, (int, float, np.integer, np.floating)):
        distance = float(value)
    elif isinstance(value, (tuple, list)):
        scalars = [
            x for x in value
            if isinstance(
                x,
                (int, float, np.integer, np.floating),
            )
        ]
        if len(scalars) == 1:
            distance = float(scalars[0])
        elif value and isinstance(
            value[0],
            (int, float, np.integer, np.floating),
        ):
            distance = float(value[0])
        else:
            raise RuntimeError(
                f"Cannot read distance from {method_name}: {value!r}"
            )
    else:
        raise RuntimeError(
            f"Unsupported {method_name} result: {value!r}"
        )

    if not np.isfinite(distance) or distance < 0:
        raise RuntimeError(
            f"Invalid {method_name} distance: {distance}"
        )

    return distance


def run_heuristic(fn, name, grid, depot, order):
    start = time.perf_counter()
    result = fn(grid, depot, order)
    runtime = time.perf_counter() - start
    return (
        coerce_distance(result, name),
        float(runtime),
    )


# ============================================================
# DQN EXECUTION
# ============================================================

def evaluate_dqn(
    env,
    agent,
    order,
    all_pick_locations,
):
    state = env.reset(order)

    if state.shape != (98,):
        raise RuntimeError(
            f"Expected state (98,), got {state.shape}."
        )

    terminated = False
    truncated = False

    start = time.perf_counter()

    while not (terminated or truncated):
        action = agent.select_action(
            state,
            eval_mode=True,
        )

        state, _, terminated, truncated, _ = env.step(action)

    runtime = time.perf_counter() - start

    items_collected, items_remaining, fraction = (
        collection_metrics(
            state,
            all_pick_locations,
            len(order),
        )
    )

    valid, invalid, invalid_rate = action_metrics(
        env.steps,
        env.travel_distance,
    )

    all_picked = items_collected == len(order)

    return {
        "completed": bool(terminated),
        "truncated": bool(truncated),
        "all_items_collected": bool(all_picked),
        "all_picked_no_finish": bool(
            all_picked and not terminated
        ),
        "distance": (
            float(env.travel_distance)
            if terminated
            else np.nan
        ),
        "partial_distance": float(env.travel_distance),
        "steps": int(env.steps),
        "valid_actions": valid,
        "invalid_actions": invalid,
        "invalid_action_rate": invalid_rate,
        "items_collected": items_collected,
        "items_remaining": items_remaining,
        "collection_fraction": fraction,
        "runtime": float(runtime),
    }


# ============================================================
# EVALUATE ONE ORDER
# ============================================================

def evaluate_order(
    env,
    agent,
    grid,
    depot,
    all_pick_locations,
    row,
):
    order = row["order"]

    start = time.perf_counter()
    exact = float(
        exact_optimal_distance(
            grid,
            depot,
            order,
        )
    )
    exact_runtime = time.perf_counter() - start

    s_shape, s_runtime = run_heuristic(
        S_SHAPE_ROUTE,
        "S-shape",
        grid,
        depot,
        order,
    )

    ret, r_runtime = run_heuristic(
        RETURN_ROUTE,
        "Return",
        grid,
        depot,
        order,
    )

    largest_gap, lg_runtime = run_heuristic(
        LARGEST_GAP_ROUTE,
        "Largest Gap",
        grid,
        depot,
        order,
    )

    tolerance = 1e-6

    for name, distance in [
        ("S-shape", s_shape),
        ("Return", ret),
        ("Largest Gap", largest_gap),
    ]:
        if distance + tolerance < exact:
            raise RuntimeError(
                f"{name}={distance} below exact={exact}."
            )

    dqn = evaluate_dqn(
        env,
        agent,
        order,
        all_pick_locations,
    )

    if (
        dqn["completed"]
        and dqn["distance"] + tolerance < exact
    ):
        raise RuntimeError(
            "Completed DQN route is below exact optimum."
        )

    if dqn["completed"]:
        dqn_gap = gap(dqn["distance"], exact)
        dqn_vs_s = improvement(dqn["distance"], s_shape)
        dqn_vs_r = improvement(dqn["distance"], ret)
        dqn_vs_lg = improvement(dqn["distance"], largest_gap)
    else:
        dqn_gap = np.nan
        dqn_vs_s = np.nan
        dqn_vs_r = np.nan
        dqn_vs_lg = np.nan

    return {
        "distribution": row["distribution"],
        "order_size": row["order_size"],
        "condition_index": row["condition_index"],
        "seed": row["seed"],
        "order": str(order),

        "exact_distance": exact,
        "exact_runtime": float(exact_runtime),

        "s_shape_distance": s_shape,
        "s_shape_gap": gap(s_shape, exact),
        "s_shape_runtime": s_runtime,

        "return_distance": ret,
        "return_gap": gap(ret, exact),
        "return_runtime": r_runtime,

        "largest_gap_distance": largest_gap,
        "largest_gap_gap": gap(largest_gap, exact),
        "largest_gap_runtime": lg_runtime,

        "dqn_completed": dqn["completed"],
        "dqn_truncated": dqn["truncated"],
        "dqn_all_items_collected": dqn[
            "all_items_collected"
        ],
        "dqn_all_picked_no_finish": dqn[
            "all_picked_no_finish"
        ],
        "dqn_distance": dqn["distance"],
        "dqn_partial_distance": dqn["partial_distance"],
        "dqn_gap": dqn_gap,
        "dqn_steps": dqn["steps"],
        "dqn_valid_actions": dqn["valid_actions"],
        "dqn_invalid_actions": dqn["invalid_actions"],
        "dqn_invalid_action_rate": dqn[
            "invalid_action_rate"
        ],
        "dqn_items_collected": dqn["items_collected"],
        "dqn_items_remaining": dqn["items_remaining"],
        "dqn_collection_fraction": dqn[
            "collection_fraction"
        ],
        "dqn_runtime": dqn["runtime"],

        "dqn_improvement_vs_s_shape": dqn_vs_s,
        "dqn_improvement_vs_return": dqn_vs_r,
        "dqn_improvement_vs_largest_gap": dqn_vs_lg,
    }


# ============================================================
# SUMMARIES
# ============================================================

def condition_summary(
    results,
    distribution,
    order_size,
):
    rows = [
        r for r in results
        if r["distribution"] == distribution
        and r["order_size"] == order_size
    ]

    completed = [
        r for r in rows
        if r["dqn_completed"]
    ]

    n = len(rows)
    completed_n = len(completed)

    return {
        "distribution": distribution,
        "order_size": order_size,
        "orders": n,

        "dqn_completed_orders": completed_n,
        "dqn_completion_rate":
            100.0 * completed_n / n,

        "dqn_any_pick_rate":
            100.0
            * sum(r["dqn_items_collected"] > 0 for r in rows)
            / n,

        "dqn_all_items_collected_rate":
            100.0
            * sum(r["dqn_all_items_collected"] for r in rows)
            / n,

        "dqn_all_picked_no_finish_rate":
            100.0
            * sum(r["dqn_all_picked_no_finish"] for r in rows)
            / n,

        "dqn_mean_collection_fraction":
            safe_mean(
                [r["dqn_collection_fraction"] for r in rows]
            ),

        "dqn_mean_invalid_action_rate":
            safe_mean(
                [r["dqn_invalid_action_rate"] for r in rows]
            ),

        "exact_mean_distance":
            safe_mean([r["exact_distance"] for r in rows]),

        "s_shape_mean_distance":
            safe_mean([r["s_shape_distance"] for r in rows]),
        "s_shape_mean_gap":
            safe_mean([r["s_shape_gap"] for r in rows]),

        "return_mean_distance":
            safe_mean([r["return_distance"] for r in rows]),
        "return_mean_gap":
            safe_mean([r["return_gap"] for r in rows]),

        "largest_gap_mean_distance":
            safe_mean(
                [r["largest_gap_distance"] for r in rows]
            ),
        "largest_gap_mean_gap":
            safe_mean([r["largest_gap_gap"] for r in rows]),

        "dqn_route_quality_n": completed_n,

        "dqn_mean_distance_completed":
            safe_mean([r["dqn_distance"] for r in completed]),

        "dqn_median_distance_completed":
            safe_median([r["dqn_distance"] for r in completed]),

        "dqn_mean_gap_completed":
            safe_mean([r["dqn_gap"] for r in completed]),

        "dqn_median_gap_completed":
            safe_median([r["dqn_gap"] for r in completed]),

        "dqn_mean_improvement_vs_s_shape_completed":
            safe_mean(
                [
                    r["dqn_improvement_vs_s_shape"]
                    for r in completed
                ]
            ),

        "dqn_mean_improvement_vs_return_completed":
            safe_mean(
                [
                    r["dqn_improvement_vs_return"]
                    for r in completed
                ]
            ),

        "dqn_mean_improvement_vs_largest_gap_completed":
            safe_mean(
                [
                    r["dqn_improvement_vs_largest_gap"]
                    for r in completed
                ]
            ),

        "exact_mean_runtime":
            safe_mean([r["exact_runtime"] for r in rows]),
        "s_shape_mean_runtime":
            safe_mean([r["s_shape_runtime"] for r in rows]),
        "return_mean_runtime":
            safe_mean([r["return_runtime"] for r in rows]),
        "largest_gap_mean_runtime":
            safe_mean([r["largest_gap_runtime"] for r in rows]),
        "dqn_mean_runtime":
            safe_mean([r["dqn_runtime"] for r in rows]),
    }


def build_method_long(results):
    output = []

    for r in results:
        common = {
            "distribution": r["distribution"],
            "order_size": r["order_size"],
            "condition_index": r["condition_index"],
            "seed": r["seed"],
        }

        for method, completed, distance, method_gap, runtime in [
            (
                "Exact",
                True,
                r["exact_distance"],
                0.0,
                r["exact_runtime"],
            ),
            (
                "S-shape",
                True,
                r["s_shape_distance"],
                r["s_shape_gap"],
                r["s_shape_runtime"],
            ),
            (
                "Return",
                True,
                r["return_distance"],
                r["return_gap"],
                r["return_runtime"],
            ),
            (
                "Largest Gap",
                True,
                r["largest_gap_distance"],
                r["largest_gap_gap"],
                r["largest_gap_runtime"],
            ),
            (
                "DQN",
                r["dqn_completed"],
                r["dqn_distance"],
                r["dqn_gap"],
                r["dqn_runtime"],
            ),
        ]:
            output.append(
                {
                    **common,
                    "method": method,
                    "completed": completed,
                    "distance": distance,
                    "optimality_gap": method_gap,
                    "runtime": runtime,
                }
            )

    return output


def build_paired_comparisons(results):
    output = []

    for r in results:
        if not r["dqn_completed"]:
            continue

        for heuristic, distance_key, improvement_key in [
            (
                "S-shape",
                "s_shape_distance",
                "dqn_improvement_vs_s_shape",
            ),
            (
                "Return",
                "return_distance",
                "dqn_improvement_vs_return",
            ),
            (
                "Largest Gap",
                "largest_gap_distance",
                "dqn_improvement_vs_largest_gap",
            ),
        ]:
            h = r[distance_key]
            d = r["dqn_distance"]

            output.append(
                {
                    "distribution": r["distribution"],
                    "order_size": r["order_size"],
                    "condition_index": r["condition_index"],
                    "seed": r["seed"],
                    "heuristic": heuristic,
                    "exact_distance": r["exact_distance"],
                    "dqn_distance": d,
                    "heuristic_distance": h,
                    "dqn_minus_heuristic_distance": d - h,
                    "dqn_improvement_percent":
                        r[improvement_key],
                    "dqn_optimality_gap": r["dqn_gap"],
                    "dqn_better_than_heuristic": d < h,
                    "same_distance":
                        math.isclose(d, h, abs_tol=1e-9),
                }
            )

    return output


# ============================================================
# FILE OUTPUT
# ============================================================

def save_csv(rows, path):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_config():
    config = {
        "run_name": RUN_NAME,
        "phase": "final_test",
        "source_model": str(SOURCE_MODEL_PATH),
        "model_frozen": True,
        "additional_training": False,
        "order_sizes": ORDER_SIZES,
        "distributions": DISTRIBUTIONS,
        "orders_per_condition": ORDERS_PER_CONDITION,
        "total_orders":
            len(ORDER_SIZES)
            * len(DISTRIBUTIONS)
            * ORDERS_PER_CONDITION,
        "final_seed_starts": {
            f"{dist}_{size}": seed
            for (dist, size), seed in FINAL_SEED_STARTS.items()
        },
        "max_steps": MAX_STEPS,
        "methods": [
            "DQN",
            "S-shape",
            "Return",
            "Largest Gap",
            "Exact",
        ],
        "optimality_gap_formula":
            "(method - exact) / exact * 100",
        "dqn_vs_heuristic_formula":
            "(heuristic - dqn) / heuristic * 100",
        "dqn_incomplete_policy":
            (
                "partial distance stored diagnostically; "
                "excluded from route-quality comparisons"
        ),
        "evaluation_seed": EVALUATION_SEED,
        "device": DEVICE,
    }

    with CONFIG_PATH.open("w") as f:
        json.dump(config, f, indent=4)


def print_summary(summary):
    print(
        "\n"
        "------------------------------------------------"
    )
    print(
        f"{summary['distribution'].upper()} "
        f"| {summary['order_size']} PICKS"
    )
    print(
        "------------------------------------------------"
    )

    print(
        f"DQN completion:        "
        f"{summary['dqn_completion_rate']:.1f}% "
        f"({summary['dqn_completed_orders']}/"
        f"{summary['orders']})"
    )

    print(
        f"Exact mean distance:   "
        f"{summary['exact_mean_distance']:.2f}"
    )

    print(
        f"S-shape:               "
        f"{summary['s_shape_mean_distance']:.2f}"
        f" | gap {summary['s_shape_mean_gap']:.2f}%"
    )

    print(
        f"Return:                "
        f"{summary['return_mean_distance']:.2f}"
        f" | gap {summary['return_mean_gap']:.2f}%"
    )

    print(
        f"Largest Gap:           "
        f"{summary['largest_gap_mean_distance']:.2f}"
        f" | gap {summary['largest_gap_mean_gap']:.2f}%"
    )

    print(
        f"DQN completed:         "
        f"{summary['dqn_mean_distance_completed']:.2f}"
        f" | gap {summary['dqn_mean_gap_completed']:.2f}%"
        f" | n={summary['dqn_route_quality_n']}"
    )

    print(
        f"DQN vs S-shape:        "
        f"{summary['dqn_mean_improvement_vs_s_shape_completed']:+.2f}%"
    )

    print(
        f"DQN vs Return:         "
        f"{summary['dqn_mean_improvement_vs_return_completed']:+.2f}%"
    )

    print(
        f"DQN vs Largest Gap:    "
        f"{summary['dqn_mean_improvement_vs_largest_gap_completed']:+.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "\n"
        "================================================\n"
        "FINAL WAREHOUSE ROUTING EVALUATION\n"
        "================================================"
    )

    set_random_seeds(EVALUATION_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_config()

    grid, depot, aisle_columns, all_pick_locations = (
        create_warehouse()
    )

    env = create_environment(
        grid,
        depot,
        all_pick_locations,
    )

    agent = create_agent(env)
    load_final_policy(agent)

    test_state = env.reset(
        all_pick_locations[:MAX_ORDER_SIZE]
    )
    test_q = agent.get_q_values(test_state)

    if test_state.shape != (98,):
        raise RuntimeError(
            f"Expected state (98,), got {test_state.shape}."
        )

    if test_q.shape != (4,):
        raise RuntimeError(
            f"Expected Q shape (4,), got {test_q.shape}."
        )

    total_orders = (
        len(ORDER_SIZES)
        * len(DISTRIBUTIONS)
        * ORDERS_PER_CONDITION
    )

    print(
        f"\nFrozen model:          mixed-size best checkpoint\n"
        f"Order sizes:           {ORDER_SIZES}\n"
        f"Distributions:         {DISTRIBUTIONS}\n"
        f"Orders/condition:      {ORDERS_PER_CONDITION}\n"
        f"Total final orders:    {total_orders}\n"
        f"DQN max steps:         {MAX_STEPS}\n"
        f"Exact benchmark:       yes\n"
        f"Device:                {DEVICE}\n"
    )

    print("Creating fresh final-test orders...")

    final_orders = create_all_final_orders(
        all_pick_locations,
        aisle_columns,
    )

    save_csv(
        [
            {
                "distribution": r["distribution"],
                "order_size": r["order_size"],
                "condition_index": r["condition_index"],
                "seed": r["seed"],
                "order": str(r["order"]),
            }
            for r in final_orders
        ],
        ORDERS_PATH,
    )

    print(f"Saved {len(final_orders)} final orders.")

    all_results = []
    summaries = []

    start_all = time.perf_counter()

    for distribution in DISTRIBUTIONS:
        for order_size in ORDER_SIZES:
            rows = [
                r for r in final_orders
                if r["distribution"] == distribution
                and r["order_size"] == order_size
            ]

            print(
                "\n"
                "================================================"
            )
            print(
                f"FINAL CONDITION: "
                f"{distribution.upper()} | {order_size} PICKS"
            )
            print(
                "================================================"
            )

            for index, row in enumerate(rows, start=1):
                result = evaluate_order(
                    env,
                    agent,
                    grid,
                    depot,
                    all_pick_locations,
                    row,
                )
                all_results.append(result)

                if (
                    index == 1
                    or index % 10 == 0
                    or index == len(rows)
                ):
                    dqn_text = (
                        f"{result['dqn_distance']:.0f}"
                        if np.isfinite(result["dqn_distance"])
                        else "NA"
                    )

                    print(
                        f"Order {index:02d}/{len(rows)}"
                        f" | opt {result['exact_distance']:.0f}"
                        f" | S {result['s_shape_distance']:.0f}"
                        f" | R {result['return_distance']:.0f}"
                        f" | LG {result['largest_gap_distance']:.0f}"
                        f" | DQN {int(result['dqn_completed'])}"
                        f" / {dqn_text}"
                        f" | collect "
                        f"{result['dqn_items_collected']}/{order_size}"
                    )

                save_csv(all_results, RAW_PATH)

            summary = condition_summary(
                all_results,
                distribution,
                order_size,
            )
            summaries.append(summary)

            save_csv(summaries, SUMMARY_PATH)
            print_summary(summary)

    method_long = build_method_long(all_results)
    paired = build_paired_comparisons(all_results)

    save_csv(method_long, LONG_PATH)
    save_csv(paired, PAIRED_PATH)

    runtime = time.perf_counter() - start_all

    print(
        "\n"
        "================================================\n"
        "FINAL EVALUATION COMPLETE\n"
        "================================================"
    )

    print(f"Total runtime: {runtime:.2f} seconds\n")

    for s in summaries:
        print(
            f"{s['distribution']:9s}"
            f" | {s['order_size']:2d} picks"
            f" | DQN complete {s['dqn_completion_rate']:5.1f}%"
            f" | DQN gap {s['dqn_mean_gap_completed']:6.2f}%"
            f" (n={s['dqn_route_quality_n']:2d})"
            f" | S {s['s_shape_mean_gap']:6.2f}%"
            f" | R {s['return_mean_gap']:6.2f}%"
            f" | LG {s['largest_gap_mean_gap']:6.2f}%"
        )

    print(
        "\nDQN distance/gap comparisons use completed DQN orders only."
    )
    print(
        "Incomplete DQN partial distance remains in final_results.csv."
    )

    print("\nSaved outputs:")
    print(f"  {CONFIG_PATH}")
    print(f"  {ORDERS_PATH}")
    print(f"  {RAW_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {LONG_PATH}")
    print(f"  {PAIRED_PATH}")


if __name__ == "__main__":
    main()
