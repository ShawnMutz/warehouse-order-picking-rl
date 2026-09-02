"""
Cross-cardinality evaluation of the trained Set-DQN.

Purpose
-------
Evaluate how far the successful five-pick curriculum policy
generalises to larger order sizes without any additional training.

The policy is frozen.

Evaluated order sizes:
    5
    10
    15
    20

Distribution:
    uniform only

This deliberately holds the order distribution fixed while changing
only order cardinality. Clustered/OOD evaluation is reserved for the
final experimental evaluation.

Source model
------------
Best checkpoint from:

    set_dqn_curriculum_five_pick_from_four_30_cycles

The five-pick checkpoint achieved 100% completion on its 50
development probes.

Evaluation design
-----------------
50 deterministic orders are evaluated at each cardinality.

Five-pick orders reproduce the existing five-pick development probe
set beginning at seed 90_000.

The 10-, 15-, and 20-pick orders use separate deterministic seed
ranges.

No learning occurs during this script.

For every order the script records:
    - completion
    - items collected
    - collection fraction
    - travel distance
    - number of steps
    - invalid-action rate
    - exact optimal distance
    - optimality gap for completed episodes
    - evaluation runtime

The Ratliff-Rosenthal exact solver is used only as an evaluation
benchmark.
"""

from utils.reward_shaping import get_remaining_pick_locations
from utils.order_generation import generate_order
from exact.ratliff_rosenthal import exact_optimal_distance
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


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================


# ============================================================
# RUN SETTINGS
# ============================================================

RUN_NAME = "set_dqn_cross_cardinality_from_five_pick"

EVALUATION_SEED = 42

ORDER_SIZES = [
    5,
    10,
    15,
    20,
]

ORDERS_PER_SIZE = 50

DISTRIBUTION = "uniform"

MAX_ORDER_SIZE = 20

MAX_STEPS = 150

DEVICE = "cpu"


# ============================================================
# SOURCE MODEL
# ============================================================

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "set_dqn_curriculum_five_pick_from_four_30_cycles"
    / "set_dqn_best_model.pt"
)


# ============================================================
# ORDER SEEDS
# ============================================================

# Five-pick seed reproduces the existing five-pick development
# probe set.
#
# Larger cardinalities receive independent deterministic
# seed ranges.

SEED_START_BY_ORDER_SIZE = {
    5: 90_000,
    10: 100_000,
    15: 110_000,
    20: 120_000,
}


EXPECTED_FIRST_FIVE_PICK_ORDER = (
    (1, 9),
    (3, 15),
    (9, 9),
    (10, 3),
    (11, 11),
)


# ============================================================
# SET-DQN SETTINGS
# ============================================================

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds(
    seed,
):
    """
    Set Python, NumPy and PyTorch random seeds.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# WAREHOUSE
# ============================================================

def create_warehouse():
    """
    Create the fixed 14 x 17 warehouse used throughout the
    project.
    """

    height = 14
    width = 17

    grid = np.ones(
        (height, width),
        dtype=np.int8,
    )

    # Front cross aisle.
    grid[0, :] = 0

    # Rear cross aisle.
    grid[13, :] = 0

    aisle_columns = [
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15,
    ]

    for col in aisle_columns:
        grid[:, col] = 0

    depot = (
        0,
        0,
    )

    all_pick_locations = [
        (row, col)
        for col in aisle_columns
        for row in range(
            1,
            13,
        )
    ]

    if len(
        all_pick_locations
    ) != 96:
        raise AssertionError(
            "Expected exactly 96 pick locations."
        )

    return (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    )


# ============================================================
# ENVIRONMENT
# ============================================================

def create_environment(
    grid,
    depot,
    all_pick_locations,
):
    """
    Create the project WarehouseEnv.
    """

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


# ============================================================
# AGENT
# ============================================================

def create_agent(
    env,
):
    """
    Create Set-DQN with the architecture used by the trained
    checkpoint.

    No training will occur in this script.
    """

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


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_policy_state_dict(
    checkpoint,
):
    """
    Extract policy-network parameters from the saved checkpoint.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Unexpected checkpoint format. "
            "Expected dictionary."
        )

    possible_keys = [
        "policy_state_dict",
        "policy_net_state_dict",
        "policy_net",
        "model_state_dict",
        "state_dict",
    ]

    for key in possible_keys:

        if key in checkpoint:

            candidate = checkpoint[
                key
            ]

            if isinstance(
                candidate,
                dict,
            ):
                return candidate

    # The checkpoint itself may be a PyTorch state_dict.
    values = list(
        checkpoint.values()
    )

    if (
        values
        and all(
            torch.is_tensor(value)
            for value in values
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Could not locate policy-network parameters.\n"
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def load_policy(
    agent,
    checkpoint_path,
):
    """
    Load the frozen five-pick curriculum policy.
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Five-pick checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print(
        "\nLoading frozen five-pick Set-DQN:"
    )

    print(
        f"  {checkpoint_path}"
    )

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

    state_dict = extract_policy_state_dict(
        checkpoint
    )

    agent.policy_net.load_state_dict(
        state_dict
    )

    agent.target_net.load_state_dict(
        state_dict
    )

    agent.policy_net.eval()

    agent.target_net.eval()

    print(
        "Frozen five-pick policy loaded successfully."
    )


# ============================================================
# ORDER KEY
# ============================================================

def order_key(
    order,
):
    """
    Canonical order-independent representation.
    """

    return tuple(
        sorted(
            tuple(location)
            for location in order
        )
    )


# ============================================================
# EVALUATION ORDERS
# ============================================================

def create_evaluation_orders(
    all_pick_locations,
    aisle_columns,
    order_size,
):
    """
    Generate deterministic unique uniform orders for one
    cardinality.
    """

    if order_size not in SEED_START_BY_ORDER_SIZE:

        raise ValueError(
            f"No seed range configured for order size "
            f"{order_size}."
        )

    seed = SEED_START_BY_ORDER_SIZE[
        order_size
    ]

    orders = []

    seen_keys = set()

    attempts = 0

    max_attempts = 100_000

    while len(
        orders
    ) < ORDERS_PER_SIZE:

        attempts += 1

        if attempts > max_attempts:

            raise RuntimeError(
                f"Unable to create {ORDERS_PER_SIZE} "
                f"unique orders of size {order_size}."
            )

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=order_size,
            distribution=DISTRIBUTION,
            seed=seed,
            aisle_columns=aisle_columns,
        )

        key = order_key(
            order
        )

        if key not in seen_keys:

            orders.append(
                list(
                    key
                )
            )

            seen_keys.add(
                key
            )

        seed += 1

    # --------------------------------------------------------
    # Five-pick fingerprint.
    # --------------------------------------------------------

    if order_size == 5:

        first_order = order_key(
            orders[0]
        )

        if (
            first_order
            != EXPECTED_FIRST_FIVE_PICK_ORDER
        ):

            raise RuntimeError(
                "Five-pick evaluation set does not match "
                "the existing development probes.\n"
                f"Expected: "
                f"{EXPECTED_FIRST_FIVE_PICK_ORDER}\n"
                f"Received: {first_order}"
            )

    return orders


# ============================================================
# COLLECTION METRICS
# ============================================================

def get_collection_metrics(
    environment_state,
    all_pick_locations,
    order_size,
):
    """
    Calculate collected/remaining item counts from raw
    WarehouseEnv state.
    """

    remaining_picks = (
        get_remaining_pick_locations(
            environment_state,
            all_pick_locations,
        )
    )

    items_remaining = len(
        remaining_picks
    )

    items_collected = (
        order_size
        - items_remaining
    )

    items_collected = max(
        0,
        min(
            order_size,
            items_collected,
        ),
    )

    items_remaining = max(
        0,
        min(
            order_size,
            items_remaining,
        ),
    )

    collection_fraction = (
        items_collected
        / order_size
    )

    return (
        items_collected,
        items_remaining,
        float(
            collection_fraction
        ),
    )


# ============================================================
# ACTION METRICS
# ============================================================

def calculate_action_metrics(
    steps,
    travel_distance,
):
    """
    Infer valid and invalid movement-action counts.
    """

    valid_actions = int(
        round(
            travel_distance
        )
    )

    invalid_actions = max(
        0,
        int(
            steps
        )
        - valid_actions,
    )

    if steps > 0:

        invalid_action_rate = (
            100.0
            * invalid_actions
            / steps
        )

    else:

        invalid_action_rate = 0.0

    return (
        valid_actions,
        invalid_actions,
        float(
            invalid_action_rate
        ),
    )


# ============================================================
# EXACT OPTIMA
# ============================================================

def calculate_exact_optima(
    grid,
    depot,
    orders,
    order_size,
):
    """
    Calculate exact optimal distance for each evaluation order.
    """

    optima = []

    print(
        f"\nCalculating exact optima "
        f"for {order_size}-pick orders..."
    )

    start_time = time.perf_counter()

    for (
        index,
        order,
    ) in enumerate(
        orders,
        start=1,
    ):

        optimum = exact_optimal_distance(
            grid,
            depot,
            order,
        )

        optimum = float(
            optimum
        )

        if optimum >= MAX_STEPS:

            raise RuntimeError(
                f"{order_size}-pick evaluation order "
                f"{index} has exact optimum "
                f"{optimum:.0f}, which reaches or exceeds "
                f"MAX_STEPS={MAX_STEPS}."
            )

        optima.append(
            optimum
        )

        if (
            index == 1
            or index % 10 == 0
            or index == len(orders)
        ):

            print(
                f"  {index:02d}/{len(orders)}"
                f" | optimum "
                f"{optimum:.0f}"
            )

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Exact {order_size}-pick calculations complete "
        f"in {runtime:.2f} seconds."
    )

    print(
        f"Minimum optimum: "
        f"{min(optima):.0f}"
    )

    print(
        f"Maximum optimum: "
        f"{max(optima):.0f}"
    )

    return optima


# ============================================================
# SINGLE ORDER EVALUATION
# ============================================================

def evaluate_order(
    env,
    agent,
    order,
    optimal_distance,
    all_pick_locations,
):
    """
    Greedily evaluate the frozen Set-DQN on one order.
    """

    environment_state = env.reset(
        order
    )

    if environment_state.shape != (
        98,
    ):
        raise RuntimeError(
            f"Expected raw state shape (98,), "
            f"received {environment_state.shape}."
        )

    terminated = False

    truncated = False

    start_time = (
        time.perf_counter()
    )

    while not (
        terminated
        or truncated
    ):

        action = agent.select_action(
            environment_state,
            eval_mode=True,
        )

        (
            next_environment_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        environment_state = (
            next_environment_state
        )

    runtime = (
        time.perf_counter()
        - start_time
    )

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        environment_state,
        all_pick_locations,
        len(
            order
        ),
    )

    (
        valid_actions,
        invalid_actions,
        invalid_action_rate,
    ) = calculate_action_metrics(
        env.steps,
        env.travel_distance,
    )

    if terminated:

        optimality_gap = (
            (
                env.travel_distance
                - optimal_distance
            )
            / optimal_distance
            * 100.0
        )

        if optimality_gap < -1e-6:

            raise RuntimeError(
                "Set-DQN route appears shorter than "
                "the exact optimal distance."
            )

    else:

        optimality_gap = np.nan

    return {
        "completed":
            bool(
                terminated
            ),

        "truncated":
            bool(
                truncated
            ),

        "items_collected":
            items_collected,

        "items_remaining":
            items_remaining,

        "collection_fraction":
            collection_fraction,

        "distance":
            float(
                env.travel_distance
            ),

        "optimal_distance":
            float(
                optimal_distance
            ),

        "optimality_gap":
            (
                float(
                    optimality_gap
                )
                if np.isfinite(
                    optimality_gap
                )
                else np.nan
            ),

        "steps":
            int(
                env.steps
            ),

        "valid_actions":
            valid_actions,

        "invalid_actions":
            invalid_actions,

        "invalid_action_rate":
            invalid_action_rate,

        "runtime":
            float(
                runtime
            ),
    }


# ============================================================
# CARDINALITY EVALUATION
# ============================================================

def evaluate_cardinality(
    env,
    agent,
    orders,
    optima,
    order_size,
    all_pick_locations,
):
    """
    Evaluate all orders for one cardinality.
    """

    raw_results = []

    print(
        "\n"
        "------------------------------------------------\n"
        f"EVALUATING FROZEN SET-DQN: "
        f"{order_size} PICKS\n"
        "------------------------------------------------"
    )

    for (
        index,
        (
            order,
            optimum,
        ),
    ) in enumerate(
        zip(
            orders,
            optima,
        ),
        start=1,
    ):

        result = evaluate_order(
            env,
            agent,
            order,
            optimum,
            all_pick_locations,
        )

        raw_results.append(
            {
                "order_size":
                    order_size,

                "order_index":
                    index,

                "distribution":
                    DISTRIBUTION,

                "order":
                    str(
                        order
                    ),

                **result,
            }
        )

        if (
            index == 1
            or index % 10 == 0
            or index == len(orders)
        ):

            print(
                f"Order "
                f"{index:02d}/{len(orders)}"
                f" | complete "
                f"{int(result['completed'])}"
                f" | collect "
                f"{result['items_collected']}"
                f"/{order_size}"
                f" | distance "
                f"{result['distance']:.0f}"
                f" | optimum "
                f"{result['optimal_distance']:.0f}"
                f" | invalid "
                f"{result['invalid_action_rate']:.1f}%"
            )

    # --------------------------------------------------------
    # Completion metrics
    # --------------------------------------------------------

    completed_results = [
        row
        for row in raw_results
        if row[
            "completed"
        ]
    ]

    completed_count = len(
        completed_results
    )

    completion_rate = (
        100.0
        * completed_count
        / len(
            raw_results
        )
    )

    any_pick_count = sum(
        row[
            "items_collected"
        ] > 0
        for row in raw_results
    )

    any_pick_rate = (
        100.0
        * any_pick_count
        / len(
            raw_results
        )
    )

    mean_items_collected = float(
        np.mean(
            [
                row[
                    "items_collected"
                ]
                for row in raw_results
            ]
        )
    )

    mean_collection_fraction = float(
        np.mean(
            [
                row[
                    "collection_fraction"
                ]
                for row in raw_results
            ]
        )
    )

    mean_invalid_action_rate = float(
        np.mean(
            [
                row[
                    "invalid_action_rate"
                ]
                for row in raw_results
            ]
        )
    )

    mean_steps = float(
        np.mean(
            [
                row[
                    "steps"
                ]
                for row in raw_results
            ]
        )
    )

    # --------------------------------------------------------
    # Route-quality metrics only for completed orders.
    # --------------------------------------------------------

    if completed_results:

        mean_completed_distance = float(
            np.mean(
                [
                    row[
                        "distance"
                    ]
                    for row in completed_results
                ]
            )
        )

        gaps = [
            row[
                "optimality_gap"
            ]
            for row in completed_results
        ]

        mean_optimality_gap = float(
            np.mean(
                gaps
            )
        )

        median_optimality_gap = float(
            np.median(
                gaps
            )
        )

        max_optimality_gap = float(
            np.max(
                gaps
            )
        )

        exact_optimal_count = sum(
            abs(
                row[
                    "optimality_gap"
                ]
            ) <= 1e-6
            for row in completed_results
        )

        exact_optimal_rate_completed = (
            100.0
            * exact_optimal_count
            / completed_count
        )

    else:

        mean_completed_distance = np.nan

        mean_optimality_gap = np.nan

        median_optimality_gap = np.nan

        max_optimality_gap = np.nan

        exact_optimal_count = 0

        exact_optimal_rate_completed = np.nan

    summary = {
        "order_size":
            order_size,

        "distribution":
            DISTRIBUTION,

        "orders":
            len(
                raw_results
            ),

        "completed_orders":
            completed_count,

        "completion_rate":
            completion_rate,

        "any_pick_count":
            any_pick_count,

        "any_pick_rate":
            any_pick_rate,

        "mean_items_collected":
            mean_items_collected,

        "mean_collection_fraction":
            mean_collection_fraction,

        "mean_invalid_action_rate":
            mean_invalid_action_rate,

        "mean_steps":
            mean_steps,

        "mean_completed_distance":
            mean_completed_distance,

        "mean_optimality_gap":
            mean_optimality_gap,

        "median_optimality_gap":
            median_optimality_gap,

        "max_optimality_gap":
            max_optimality_gap,

        "exact_optimal_count":
            exact_optimal_count,

        "exact_optimal_rate_completed":
            exact_optimal_rate_completed,
    }

    return (
        summary,
        raw_results,
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    rows,
    output_path,
):
    """
    Save dictionary records to CSV.
    """

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
            fieldnames=rows[
                0
            ].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "================================================\n"
        "SET-DQN CROSS-CARDINALITY EVALUATION\n"
        "================================================"
    )

    set_random_seeds(
        EVALUATION_SEED
    )

    # ========================================================
    # WAREHOUSE
    # ========================================================

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    ) = create_warehouse()

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = create_environment(
        grid,
        depot,
        all_pick_locations,
    )

    # ========================================================
    # AGENT
    # ========================================================

    agent = create_agent(
        env
    )

    load_policy(
        agent,
        SOURCE_MODEL_PATH,
    )

    # ========================================================
    # BASIC NETWORK TEST
    # ========================================================

    test_order = all_pick_locations[
        :5
    ]

    test_state = env.reset(
        test_order
    )

    test_q_values = agent.get_q_values(
        test_state
    )

    if test_state.shape != (
        98,
    ):
        raise RuntimeError(
            f"Expected state shape (98,), "
            f"received {test_state.shape}."
        )

    if test_q_values.shape != (
        4,
    ):
        raise RuntimeError(
            f"Expected four Q-values, "
            f"received {test_q_values.shape}."
        )

    print(
        f"\nRun name:               {RUN_NAME}\n"
        f"Model:                  five-pick curriculum best\n"
        f"Policy frozen:          yes\n"
        f"Training during eval:   no\n"
        f"Distribution:           {DISTRIBUTION}\n"
        f"Order sizes:            {ORDER_SIZES}\n"
        f"Orders per size:        {ORDERS_PER_SIZE}\n"
        f"Total evaluation orders:"
        f" {len(ORDER_SIZES) * ORDERS_PER_SIZE}\n"
        f"Max episode steps:      {MAX_STEPS}\n"
        f"Raw state dimension:    {test_state.shape[0]}\n"
        f"Max model order size:   {MAX_ORDER_SIZE}\n"
        f"Device:                 {DEVICE}\n"
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

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
        / "cross_cardinality_summary.csv"
    )

    raw_path = (
        results_directory
        / "cross_cardinality_raw.csv"
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {
        "run_name":
            RUN_NAME,

        "evaluation_type":
            "frozen_cross_cardinality",

        "source_model":
            str(
                SOURCE_MODEL_PATH
            ),

        "source_training_order_size":
            5,

        "policy_frozen":
            True,

        "additional_training":
            False,

        "order_sizes":
            ORDER_SIZES,

        "orders_per_size":
            ORDERS_PER_SIZE,

        "distribution":
            DISTRIBUTION,

        "seed_start_by_order_size":
            SEED_START_BY_ORDER_SIZE,

        "evaluation_seed":
            EVALUATION_SEED,

        "max_steps":
            MAX_STEPS,

        "max_order_size":
            MAX_ORDER_SIZE,

        "pick_embedding_dim":
            PICK_EMBEDDING_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "device":
            DEVICE,
    }

    with config_path.open(
        "w"
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    all_order_rows = []

    summaries = []

    raw_results = []

    total_start_time = (
        time.perf_counter()
    )

    for order_size in ORDER_SIZES:

        print(
            "\n"
            "================================================"
        )

        print(
            f"PREPARING {order_size}-PICK EVALUATION"
        )

        print(
            "================================================"
        )

        # ----------------------------------------------------
        # Generate evaluation orders.
        # ----------------------------------------------------

        orders = create_evaluation_orders(
            all_pick_locations,
            aisle_columns,
            order_size,
        )

        if len(
            orders
        ) != ORDERS_PER_SIZE:

            raise AssertionError(
                f"Expected {ORDERS_PER_SIZE} orders for "
                f"size {order_size}."
            )

        print(
            f"Created {len(orders)} unique "
            f"{order_size}-pick {DISTRIBUTION} orders."
        )

        # ----------------------------------------------------
        # Exact optimum.
        # ----------------------------------------------------

        optima = calculate_exact_optima(
            grid,
            depot,
            orders,
            order_size,
        )

        # ----------------------------------------------------
        # Store order definitions.
        # ----------------------------------------------------

        for (
            index,
            (
                order,
                optimum,
            ),
        ) in enumerate(
            zip(
                orders,
                optima,
            ),
            start=1,
        ):

            all_order_rows.append(
                {
                    "order_size":
                        order_size,

                    "order_index":
                        index,

                    "distribution":
                        DISTRIBUTION,

                    "seed_start":
                        SEED_START_BY_ORDER_SIZE[
                            order_size
                        ],

                    "order":
                        str(
                            order
                        ),

                    "optimal_distance":
                        optimum,
                }
            )

        # ----------------------------------------------------
        # Frozen policy evaluation.
        # ----------------------------------------------------

        (
            summary,
            size_raw_results,
        ) = evaluate_cardinality(
            env,
            agent,
            orders,
            optima,
            order_size,
            all_pick_locations,
        )

        summaries.append(
            summary
        )

        raw_results.extend(
            size_raw_results
        )

        # ----------------------------------------------------
        # Size summary.
        # ----------------------------------------------------

        gap = summary[
            "mean_optimality_gap"
        ]

        gap_text = (
            f"{gap:.2f}%"
            if np.isfinite(
                gap
            )
            else "N/A"
        )

        print(
            "\n"
            f"{order_size}-PICK SUMMARY\n"
            "------------------------------------------------"
        )

        print(
            f"Completion:          "
            f"{summary['completion_rate']:.1f}% "
            f"({summary['completed_orders']}/"
            f"{summary['orders']})"
        )

        print(
            f"Any-pick rate:       "
            f"{summary['any_pick_rate']:.1f}%"
        )

        print(
            f"Mean items:          "
            f"{summary['mean_items_collected']:.3f}"
            f"/{order_size}"
        )

        print(
            f"Collection fraction: "
            f"{summary['mean_collection_fraction']:.3f}"
        )

        print(
            f"Invalid-action rate: "
            f"{summary['mean_invalid_action_rate']:.1f}%"
        )

        print(
            f"Mean optimality gap: "
            f"{gap_text}"
        )

        print(
            f"Exactly optimal:     "
            f"{summary['exact_optimal_count']}/"
            f"{summary['completed_orders']}"
        )

        # ----------------------------------------------------
        # Persist after every cardinality.
        # ----------------------------------------------------

        save_csv(
            all_order_rows,
            orders_path,
        )

        save_csv(
            summaries,
            summary_path,
        )

        save_csv(
            raw_results,
            raw_path,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_runtime = (
        time.perf_counter()
        - total_start_time
    )

    print(
        "\n"
        "================================================\n"
        "CROSS-CARDINALITY EVALUATION COMPLETE\n"
        "================================================"
    )

    print(
        f"Total evaluation runtime: "
        f"{total_runtime:.2f} seconds\n"
    )

    print(
        "Frozen-policy scaling summary:"
    )

    for summary in summaries:

        gap = summary[
            "mean_optimality_gap"
        ]

        gap_text = (
            f"{gap:.2f}%"
            if np.isfinite(
                gap
            )
            else "N/A"
        )

        print(
            f"  {summary['order_size']:2d} picks"
            f" | complete "
            f"{summary['completion_rate']:5.1f}%"
            f" | collection "
            f"{summary['mean_collection_fraction']:.3f}"
            f" | invalid "
            f"{summary['mean_invalid_action_rate']:5.1f}%"
            f" | gap "
            f"{gap_text}"
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
        f"  Summary:\n"
        f"    {summary_path}"
    )

    print(
        f"  Raw results:\n"
        f"    {raw_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
