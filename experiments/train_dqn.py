"""
DQN one-pick learnability training.

Purpose
-------
Determine whether the movement-level DQN can learn the simplest
warehouse-routing task before progressing to larger orders.

Training:
    - 1-pick orders
    - uniform distribution
    - all 96 pick locations may appear during training

Development evaluation:
    - fixed 1-pick uniform orders
    - greedy policy (epsilon = 0)
    - no replay-buffer insertion
    - no gradient updates
    - comparison against exact optimal distance

Diagnostics:
    - items collected
    - items remaining
    - fraction of order collected
    - completion rate
    - reward
    - travel distance
    - loss
    - epsilon

This is a DEVELOPMENT experiment, not the final experiment.
"""

from utils.order_generation import generate_order
from exact.ratliff_rosenthal import exact_optimal_distance
from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Project imports MUST come after PROJECT_ROOT is added.


# ============================================================
# DEVELOPMENT SETTINGS
# ============================================================

RUN_NAME = "one_pick_learnability"

TRAIN_SEED = 42

TRAIN_EPISODES = 500

# Simplest possible routing problem.
ORDER_SIZE = 1

TRAIN_DISTRIBUTION = "uniform"

# Reduced from 500 because one-pick optimal routes are
# substantially shorter.
MAX_STEPS = 150


# ============================================================
# DEVELOPMENT EVALUATION SETTINGS
# ============================================================

DEV_ORDERS = 20

DEV_SEED_START = 50_000

EVALUATE_EVERY = 50


# ============================================================
# DQN HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 1e-4

GAMMA = 0.99

EPSILON_START = 1.0

EPSILON_MIN = 0.05

# Epsilon decays after each training update, so this is
# deliberately slow for the learnability experiment.
EPSILON_DECAY = 0.999995

BATCH_SIZE = 64

BUFFER_CAPACITY = 50_000

TARGET_UPDATE = 1_000

HIDDEN_DIM = 256


# ============================================================
# WAREHOUSE
# ============================================================

def create_warehouse():
    """
    Create the fixed 14 x 17 project warehouse.

    Layout:
        front cross aisle = row 0
        rear cross aisle = row 13

        picking aisle columns:
            1, 3, 5, 7, 9, 11, 13, 15

        storage rows:
            1..12

        depot:
            (0, 0)

    Returns
    -------
    tuple
        grid,
        depot,
        aisle_columns,
        all_pick_locations
    """

    height = 14
    width = 17

    grid = np.ones(
        (height, width),
        dtype=np.int8,
    )

    # Front cross aisle
    grid[0, :] = 0

    # Rear cross aisle
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

    # Vertical picking aisles
    for col in aisle_columns:
        grid[:, col] = 0

    depot = (0, 0)

    # 8 aisles x 12 positions = 96 fixed pick locations.
    all_pick_locations = [
        (row, col)
        for col in aisle_columns
        for row in range(1, 13)
    ]

    return (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds(seed):
    """
    Set Python, NumPy and PyTorch random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# ENVIRONMENT
# ============================================================

def create_environment(
    grid,
    depot,
    all_pick_locations,
):
    """
    Create the warehouse RL environment.
    """

    return WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=MAX_STEPS,

        # Keep reward design unchanged for this diagnostic.
        move_cost=-1.0,
        invalid_penalty=-2.0,
        pick_reward=2.0,
        completion_reward=20.0,
    )


# ============================================================
# DQN AGENT
# ============================================================

def create_agent(env):
    """
    Create the development DQN.
    """

    return DQNAgent(
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
    )


# ============================================================
# ORDER HELPERS
# ============================================================

def order_key(order):
    """
    Convert an order into a deterministic hashable form.
    """

    return tuple(
        sorted(order)
    )


def create_development_orders(
    all_pick_locations,
    aisle_columns,
):
    """
    Create a fixed development evaluation set.

    Important
    ---------
    The exact same one-location orders MAY also occur during
    training in this learnability diagnostic.

    The development set is still useful because evaluation
    itself is greedy and produces no replay-buffer updates.
    """

    orders = []

    seen = set()

    seed = DEV_SEED_START

    while len(orders) < DEV_ORDERS:

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=ORDER_SIZE,
            distribution="uniform",
            seed=seed,
            aisle_columns=aisle_columns,
        )

        key = order_key(order)

        if key not in seen:

            orders.append(order)

            seen.add(key)

        seed += 1

    return orders


def generate_training_order(
    episode,
    all_pick_locations,
    aisle_columns,
):
    """
    Generate one deterministic training order.

    For this one-pick learnability experiment, all 96
    locations are eligible during training.

    No development-location exclusion is performed.
    """

    seed = TRAIN_SEED + episode

    return generate_order(
        all_pick_locations=all_pick_locations,
        size=ORDER_SIZE,
        distribution=TRAIN_DISTRIBUTION,
        seed=seed,
        aisle_columns=aisle_columns,
    )


# ============================================================
# COLLECTION DIAGNOSTICS
# ============================================================

def get_collection_metrics(
    state,
    order_size,
):
    """
    Calculate collection progress from the environment state.

    State layout:
        state[0]  = normalised picker row
        state[1]  = normalised picker column
        state[2:] = 96 remaining-pick indicators

    A remaining-pick value greater than 0.5 is treated as
    an item that still needs to be collected.

    Returns
    -------
    tuple
        items_collected,
        items_remaining,
        collection_fraction
    """

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    remaining_bits = state[2:]

    items_remaining = int(
        np.count_nonzero(
            remaining_bits > 0.5
        )
    )

    items_collected = (
        order_size
        - items_remaining
    )

    # Defensive bounds.
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

    if order_size > 0:

        collection_fraction = (
            items_collected
            / order_size
        )

    else:

        collection_fraction = 1.0

    return (
        items_collected,
        items_remaining,
        float(collection_fraction),
    )


# ============================================================
# TRAIN ONE EPISODE
# ============================================================

def train_episode(
    env,
    agent,
    order,
):
    """
    Run one complete DQN training episode.

    Returns
    -------
    dict
        Episode-level metrics.
    """

    state = env.reset(order)

    terminated = False
    truncated = False

    total_reward = 0.0

    losses = []

    while not (
        terminated
        or truncated
    ):

        # ----------------------------------------------------
        # Action selection
        # ----------------------------------------------------

        action = agent.select_action(
            state
        )

        # ----------------------------------------------------
        # Environment transition
        # ----------------------------------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        # ----------------------------------------------------
        # Replay buffer
        # ----------------------------------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )

        # ----------------------------------------------------
        # Gradient update
        # ----------------------------------------------------

        loss = agent.train_step()

        if loss is not None:

            if not np.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite DQN loss detected: {loss}"
                )

            losses.append(
                float(loss)
            )

        total_reward += reward

        state = next_state

    # --------------------------------------------------------
    # Collection diagnostics
    # --------------------------------------------------------

    order_size = len(order)

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        state,
        order_size,
    )

    average_loss = (
        float(
            np.mean(losses)
        )
        if losses
        else np.nan
    )

    return {
        "reward":
            float(total_reward),

        "distance":
            float(env.travel_distance),

        "steps":
            int(env.steps),

        "completed":
            bool(terminated),

        "truncated":
            bool(truncated),

        "epsilon":
            float(agent.epsilon),

        "average_loss":
            average_loss,

        "training_updates":
            len(losses),

        "order_size":
            order_size,

        "items_collected":
            items_collected,

        "items_remaining":
            items_remaining,

        "collection_fraction":
            collection_fraction,
    }


# ============================================================
# GREEDY EVALUATION
# ============================================================

def evaluate_dqn_order(
    env,
    agent,
    order,
):
    """
    Evaluate the DQN greedily on one order.

    During evaluation:
        epsilon = 0
        no replay-buffer insertion
        no gradient update
    """

    previous_epsilon = (
        agent.epsilon
    )

    agent.epsilon = 0.0

    state = env.reset(order)

    terminated = False
    truncated = False

    start_time = (
        time.perf_counter()
    )

    try:

        while not (
            terminated
            or truncated
        ):

            action = agent.select_action(
                state
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            state = next_state

    finally:

        # Restore training exploration rate.
        agent.epsilon = (
            previous_epsilon
        )

    runtime = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Collection diagnostics
    # --------------------------------------------------------

    order_size = len(order)

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        state,
        order_size,
    )

    return {
        "completed":
            bool(terminated),

        "truncated":
            bool(truncated),

        "distance":
            float(env.travel_distance),

        "steps":
            int(env.steps),

        "runtime":
            float(runtime),

        "order_size":
            order_size,

        "items_collected":
            items_collected,

        "items_remaining":
            items_remaining,

        "collection_fraction":
            collection_fraction,
    }


# ============================================================
# DEVELOPMENT OPTIMA
# ============================================================

def calculate_development_optima(
    grid,
    depot,
    development_orders,
):
    """
    Calculate exact optimal distances for every fixed
    development order.
    """

    optima = []

    print(
        "\nCalculating development-set optimal distances..."
    )

    for index, order in enumerate(
        development_orders,
        start=1,
    ):

        optimum = exact_optimal_distance(
            grid,
            depot,
            order,
        )

        optima.append(
            float(optimum)
        )

        print(
            f"  Dev order "
            f"{index:02d}/{len(development_orders)}"
            f" -> {order}"
            f" -> optimum {optimum:.0f}"
        )

    return optima


# ============================================================
# DEVELOPMENT SET EVALUATION
# ============================================================

def evaluate_development_set(
    env,
    agent,
    development_orders,
    development_optima,
    episode,
):
    """
    Evaluate the greedy DQN on all fixed development orders.
    """

    results = []

    for (
        order_index,
        (order, optimum),
    ) in enumerate(
        zip(
            development_orders,
            development_optima,
        ),
        start=1,
    ):

        evaluation = evaluate_dqn_order(
            env,
            agent,
            order,
        )

        completed = evaluation[
            "completed"
        ]

        # Optimality gap only makes sense for successful
        # complete routes.
        if completed:

            optimality_gap = (
                (
                    evaluation["distance"]
                    - optimum
                )
                / optimum
                * 100.0
            )

        else:

            optimality_gap = np.nan

        results.append(
            {
                "episode":
                    episode,

                "order_index":
                    order_index,

                "order":
                    str(
                        sorted(order)
                    ),

                "completed":
                    completed,

                "truncated":
                    evaluation[
                        "truncated"
                    ],

                "distance":
                    evaluation[
                        "distance"
                    ],

                "optimal_distance":
                    optimum,

                "optimality_gap":
                    optimality_gap,

                "steps":
                    evaluation[
                        "steps"
                    ],

                "runtime":
                    evaluation[
                        "runtime"
                    ],

                "order_size":
                    evaluation[
                        "order_size"
                    ],

                "items_collected":
                    evaluation[
                        "items_collected"
                    ],

                "items_remaining":
                    evaluation[
                        "items_remaining"
                    ],

                "collection_fraction":
                    evaluation[
                        "collection_fraction"
                    ],
            }
        )

    # --------------------------------------------------------
    # Completion metrics
    # --------------------------------------------------------

    completed_results = [
        result
        for result in results
        if result["completed"]
    ]

    completion_rate = (
        100.0
        * len(completed_results)
        / len(results)
    )

    # --------------------------------------------------------
    # Collection metrics
    # --------------------------------------------------------

    orders_collecting_any_pick = sum(
        result["items_collected"] > 0
        for result in results
    )

    any_pick_rate = (
        100.0
        * orders_collecting_any_pick
        / len(results)
    )

    mean_items_collected = float(
        np.mean(
            [
                result["items_collected"]
                for result in results
            ]
        )
    )

    mean_collection_fraction = float(
        np.mean(
            [
                result["collection_fraction"]
                for result in results
            ]
        )
    )

    # --------------------------------------------------------
    # Completed-order metrics
    # --------------------------------------------------------

    if completed_results:

        mean_distance = float(
            np.mean(
                [
                    result["distance"]
                    for result
                    in completed_results
                ]
            )
        )

        mean_gap = float(
            np.mean(
                [
                    result["optimality_gap"]
                    for result
                    in completed_results
                ]
            )
        )

        mean_steps = float(
            np.mean(
                [
                    result["steps"]
                    for result
                    in completed_results
                ]
            )
        )

    else:

        mean_distance = np.nan
        mean_gap = np.nan
        mean_steps = np.nan

    summary = {
        "episode":
            episode,

        "completion_rate":
            completion_rate,

        "completed_orders":
            len(completed_results),

        "total_orders":
            len(results),

        "orders_collecting_any_pick":
            orders_collecting_any_pick,

        "any_pick_rate":
            any_pick_rate,

        "mean_items_collected":
            mean_items_collected,

        "mean_collection_fraction":
            mean_collection_fraction,

        "mean_completed_distance":
            mean_distance,

        "mean_optimality_gap":
            mean_gap,

        "mean_completed_steps":
            mean_steps,
    }

    return (
        summary,
        results,
    )


# ============================================================
# CSV SAVING
# ============================================================

def save_csv(
    rows,
    output_path,
):
    """
    Save list-of-dictionary results to CSV.
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
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# BEST MODEL SELECTION
# ============================================================

def development_score_is_better(
    new_summary,
    best_summary,
):
    """
    Compare development checkpoints.

    Priority:
        1. higher completion rate
        2. higher collection fraction
        3. lower optimality gap
    """

    if best_summary is None:
        return True

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    new_completion = new_summary[
        "completion_rate"
    ]

    best_completion = best_summary[
        "completion_rate"
    ]

    if new_completion > best_completion:
        return True

    if new_completion < best_completion:
        return False

    # --------------------------------------------------------
    # Collection fraction
    # --------------------------------------------------------

    new_collection = new_summary[
        "mean_collection_fraction"
    ]

    best_collection = best_summary[
        "mean_collection_fraction"
    ]

    if new_collection > best_collection:
        return True

    if new_collection < best_collection:
        return False

    # --------------------------------------------------------
    # Optimality gap
    # --------------------------------------------------------

    new_gap = new_summary[
        "mean_optimality_gap"
    ]

    best_gap = best_summary[
        "mean_optimality_gap"
    ]

    if (
        np.isnan(new_gap)
        and np.isnan(best_gap)
    ):
        return False

    if np.isnan(new_gap):
        return False

    if np.isnan(best_gap):
        return True

    return (
        new_gap
        < best_gap
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "============================================\n"
        "DQN ONE-PICK LEARNABILITY TRAINING\n"
        "============================================"
    )

    print(
        f"Run name:              {RUN_NAME}\n"
        f"Training episodes:     {TRAIN_EPISODES}\n"
        f"Training order size:   {ORDER_SIZE}\n"
        f"Training distribution: {TRAIN_DISTRIBUTION}\n"
        f"Development orders:    {DEV_ORDERS}\n"
        f"Evaluate every:        {EVALUATE_EVERY} episodes\n"
        f"Max episode steps:     {MAX_STEPS}\n"
        f"Training seed:         {TRAIN_SEED}\n"
        f"Epsilon decay:         {EPSILON_DECAY}\n"
    )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_random_seeds(
        TRAIN_SEED
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

    print(
        f"State dimension: "
        f"{2 + len(all_pick_locations)}"
    )

    print(
        f"Fixed pick locations: "
        f"{len(all_pick_locations)}"
    )

    # ========================================================
    # ENVIRONMENT + AGENT
    # ========================================================

    env = create_environment(
        grid,
        depot,
        all_pick_locations,
    )

    agent = create_agent(
        env
    )

    # ========================================================
    # DEVELOPMENT SET
    # ========================================================

    development_orders = (
        create_development_orders(
            all_pick_locations,
            aisle_columns,
        )
    )

    # IMPORTANT:
    # There is deliberately NO development_order_keys block.
    #
    # Development locations are allowed to appear in training
    # for this one-pick learnability diagnostic.

    development_optima = (
        calculate_development_optima(
            grid,
            depot,
            development_orders,
        )
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

    training_history_path = (
        results_directory
        / "dqn_training_history.csv"
    )

    development_summary_path = (
        results_directory
        / "dqn_dev_summary.csv"
    )

    development_raw_path = (
        results_directory
        / "dqn_dev_raw.csv"
    )

    best_model_path = (
        results_directory
        / "dqn_best_dev_model.pt"
    )

    final_model_path = (
        results_directory
        / "dqn_final_dev_model.pt"
    )

    # ========================================================
    # HISTORY CONTAINERS
    # ========================================================

    training_history = []

    development_summaries = []

    development_raw_results = []

    rolling_completion = deque(
        maxlen=50
    )

    rolling_collection_fraction = deque(
        maxlen=50
    )

    rolling_items_collected = deque(
        maxlen=50
    )

    rolling_rewards = deque(
        maxlen=50
    )

    best_development_summary = None

    # ========================================================
    # INITIAL UNTRAINED EVALUATION
    # ========================================================

    print(
        "\nEvaluating untrained policy..."
    )

    (
        initial_summary,
        initial_raw,
    ) = evaluate_development_set(
        env,
        agent,
        development_orders,
        development_optima,
        episode=0,
    )

    development_summaries.append(
        initial_summary
    )

    development_raw_results.extend(
        initial_raw
    )

    best_development_summary = (
        initial_summary.copy()
    )

    # Always keep an initial checkpoint.
    agent.save(
        best_model_path
    )

    print(
        f"Episode 0"
        f" | dev completion "
        f"{initial_summary['completion_rate']:.1f}%"
        f" | any-pick rate "
        f"{initial_summary['any_pick_rate']:.1f}%"
        f" | mean collection "
        f"{initial_summary['mean_collection_fraction']:.3f}"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    training_start_time = (
        time.perf_counter()
    )

    for episode in range(
        1,
        TRAIN_EPISODES + 1,
    ):

        # ----------------------------------------------------
        # Generate training order
        # ----------------------------------------------------

        order = generate_training_order(
            episode,
            all_pick_locations,
            aisle_columns,
        )

        # ----------------------------------------------------
        # Train one episode
        # ----------------------------------------------------

        metrics = train_episode(
            env,
            agent,
            order,
        )

        # ----------------------------------------------------
        # Rolling statistics
        # ----------------------------------------------------

        rolling_completion.append(
            int(
                metrics["completed"]
            )
        )

        rolling_collection_fraction.append(
            metrics[
                "collection_fraction"
            ]
        )

        rolling_items_collected.append(
            metrics[
                "items_collected"
            ]
        )

        rolling_rewards.append(
            metrics[
                "reward"
            ]
        )

        rolling_completion_rate = (
            100.0
            * float(
                np.mean(
                    rolling_completion
                )
            )
        )

        rolling_mean_collection = float(
            np.mean(
                rolling_collection_fraction
            )
        )

        rolling_mean_items = float(
            np.mean(
                rolling_items_collected
            )
        )

        rolling_reward = float(
            np.mean(
                rolling_rewards
            )
        )

        # ----------------------------------------------------
        # Store training history
        # ----------------------------------------------------

        training_row = {
            "episode":
                episode,

            "order":
                str(
                    sorted(order)
                ),

            "reward":
                metrics[
                    "reward"
                ],

            "distance":
                metrics[
                    "distance"
                ],

            "steps":
                metrics[
                    "steps"
                ],

            "completed":
                metrics[
                    "completed"
                ],

            "truncated":
                metrics[
                    "truncated"
                ],

            "order_size":
                metrics[
                    "order_size"
                ],

            "items_collected":
                metrics[
                    "items_collected"
                ],

            "items_remaining":
                metrics[
                    "items_remaining"
                ],

            "collection_fraction":
                metrics[
                    "collection_fraction"
                ],

            "epsilon":
                metrics[
                    "epsilon"
                ],

            "average_loss":
                metrics[
                    "average_loss"
                ],

            "training_updates":
                metrics[
                    "training_updates"
                ],

            "rolling_50_completion_rate":
                rolling_completion_rate,

            "rolling_50_mean_items_collected":
                rolling_mean_items,

            "rolling_50_collection_fraction":
                rolling_mean_collection,

            "rolling_50_reward":
                rolling_reward,
        }

        training_history.append(
            training_row
        )

        # ----------------------------------------------------
        # Console progress
        # ----------------------------------------------------

        if (
            episode == 1
            or episode % 10 == 0
        ):

            loss_text = (
                f"{metrics['average_loss']:.4f}"
                if np.isfinite(
                    metrics[
                        "average_loss"
                    ]
                )
                else "N/A"
            )

            print(
                f"Episode "
                f"{episode:03d}/{TRAIN_EPISODES}"
                f" | reward "
                f"{metrics['reward']:8.1f}"
                f" | steps "
                f"{metrics['steps']:3d}"
                f" | collected "
                f"{metrics['items_collected']}"
                f"/{metrics['order_size']}"
                f" | complete "
                f"{int(metrics['completed'])}"
                f" | roll collect "
                f"{rolling_mean_collection:5.2f}"
                f" | roll complete "
                f"{rolling_completion_rate:5.1f}%"
                f" | eps "
                f"{metrics['epsilon']:.3f}"
                f" | loss "
                f"{loss_text}"
            )

        # ====================================================
        # PERIODIC DEVELOPMENT EVALUATION
        # ====================================================

        if (
            episode
            % EVALUATE_EVERY
            == 0
        ):

            (
                dev_summary,
                dev_raw,
            ) = evaluate_development_set(
                env,
                agent,
                development_orders,
                development_optima,
                episode,
            )

            development_summaries.append(
                dev_summary
            )

            development_raw_results.extend(
                dev_raw
            )

            gap_value = dev_summary[
                "mean_optimality_gap"
            ]

            gap_text = (
                f"{gap_value:.2f}%"
                if np.isfinite(
                    gap_value
                )
                else "N/A"
            )

            print(
                "\n"
                "--------------------------------------------"
            )

            print(
                f"DEVELOPMENT EVALUATION "
                f"AT EPISODE {episode}"
            )

            print(
                "--------------------------------------------"
            )

            print(
                f"Completion rate:       "
                f"{dev_summary['completion_rate']:.1f}%"
            )

            print(
                f"Completed orders:      "
                f"{dev_summary['completed_orders']}"
                f"/{dev_summary['total_orders']}"
            )

            print(
                f"Any-pick rate:         "
                f"{dev_summary['any_pick_rate']:.1f}%"
            )

            print(
                f"Mean items collected:  "
                f"{dev_summary['mean_items_collected']:.3f}"
            )

            print(
                f"Mean collection frac:  "
                f"{dev_summary['mean_collection_fraction']:.3f}"
            )

            print(
                f"Mean completed dist:   "
                f"{dev_summary['mean_completed_distance']}"
            )

            print(
                f"Mean optimality gap:   "
                f"{gap_text}"
            )

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            if development_score_is_better(
                dev_summary,
                best_development_summary,
            ):

                best_development_summary = (
                    dev_summary.copy()
                )

                agent.save(
                    best_model_path
                )

                print(
                    "New best development model saved."
                )

            print()

            # ------------------------------------------------
            # Intermediate result save
            # ------------------------------------------------

            save_csv(
                training_history,
                training_history_path,
            )

            save_csv(
                development_summaries,
                development_summary_path,
            )

            save_csv(
                development_raw_results,
                development_raw_path,
            )

    # ========================================================
    # END TRAINING
    # ========================================================

    total_training_time = (
        time.perf_counter()
        - training_start_time
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    agent.save(
        final_model_path
    )

    # ========================================================
    # FINAL CSV SAVE
    # ========================================================

    save_csv(
        training_history,
        training_history_path,
    )

    save_csv(
        development_summaries,
        development_summary_path,
    )

    save_csv(
        development_raw_results,
        development_raw_path,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        "============================================\n"
        "ONE-PICK LEARNABILITY RUN COMPLETE\n"
        "============================================"
    )

    print(
        f"Training time: "
        f"{total_training_time:.2f} seconds"
    )

    print(
        f"Final epsilon: "
        f"{agent.epsilon:.4f}"
    )

    print(
        f"Final rolling completion: "
        f"{100.0 * np.mean(rolling_completion):.1f}%"
    )

    print(
        f"Final rolling collection fraction: "
        f"{np.mean(rolling_collection_fraction):.3f}"
    )

    print(
        f"Final rolling items collected: "
        f"{np.mean(rolling_items_collected):.3f}"
    )

    # ========================================================
    # BEST DEVELOPMENT CHECKPOINT
    # ========================================================

    if best_development_summary is not None:

        print(
            f"Best dev episode: "
            f"{best_development_summary['episode']}"
        )

        print(
            f"Best dev completion: "
            f"{best_development_summary['completion_rate']:.1f}%"
        )

        print(
            f"Best dev any-pick rate: "
            f"{best_development_summary['any_pick_rate']:.1f}%"
        )

        print(
            f"Best dev mean collection fraction: "
            f"{best_development_summary['mean_collection_fraction']:.3f}"
        )

        gap = best_development_summary[
            "mean_optimality_gap"
        ]

        if np.isfinite(gap):

            print(
                f"Best dev mean optimality gap: "
                f"{gap:.2f}%"
            )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Training history:\n"
        f"    {training_history_path}"
    )

    print(
        f"  Development summary:\n"
        f"    {development_summary_path}"
    )

    print(
        f"  Development raw results:\n"
        f"    {development_raw_path}"
    )

    print(
        f"  Best development model:\n"
        f"    {best_model_path}"
    )

    print(
        f"  Final development model:\n"
        f"    {final_model_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
