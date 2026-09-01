"""
Balanced one-pick Set-DQN diagnostic.

Purpose
-------
Validate that the new permutation-invariant Set-DQN can learn the
basic warehouse navigation problem before progressing to multi-pick
orders.

This is a development diagnostic, not a final-test experiment.

Training design
---------------
The warehouse contains 96 possible pick locations.

Each training cycle visits every location exactly once:

    96 episodes per cycle

For:

    15 cycles

giving:

    1440 total training episodes

and:

    15 training exposures per location.

The location order is shuffled independently and deterministically
within each cycle.

Evaluation
----------
After every complete training cycle, the greedy DQN is evaluated on
ALL 96 possible one-pick orders.

Because all locations are also used during training, this experiment
does NOT test unseen-location generalisation.

Its purpose is to verify:

    1. the Set-DQN architecture can learn navigation,
    2. one-pick performance remains strong after replacing the
       previous fixed coordinate-slot representation,
    3. successful routes can be compared with exact optimal distance.

State representation
--------------------
The Set-DQN consumes the original 98-dimensional WarehouseEnv state
directly:

    2 normalised picker coordinates
    +
    96 remaining-pick indicators

No CoordinateStateEncoder is used.

Reward shaping
--------------
Uses the same potential-based multi-pick shaping implementation in:

    utils/reward_shaping.py

For a one-pick order this reduces to:

    -(distance picker -> pick + distance pick -> depot)

before collection, and:

    -distance picker -> depot

after collection.

The exact optimiser is used only for evaluation.
"""

from utils.reward_shaping import (
    build_distance_lookup,
    calculate_potential_shaping_reward,
    get_remaining_pick_locations,
)
from exact.ratliff_rosenthal import exact_optimal_distance
from environment.warehouse_env import WarehouseEnv
from agents.set_dqn_agent import SetDQNAgent
import csv
import json
import random
import sys
import time

from collections import Counter, deque
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
# EXPERIMENT SETTINGS
# ============================================================

RUN_NAME = "set_dqn_balanced_one_pick_15_cycles"

TRAIN_SEED = 42

TRAIN_CYCLES = 15

ORDER_SIZE = 1

MAX_ORDER_SIZE = 20

MAX_STEPS = 150


# ============================================================
# EVALUATION
# ============================================================

# Evaluate after every complete balanced cycle.
EVALUATE_EVERY_CYCLES = 1


# ============================================================
# TRAINING METRICS
# ============================================================

# One complete balanced cycle.
ROLLING_WINDOW = 96

# Two progress messages per cycle.
PRINT_EVERY = 48


# ============================================================
# DQN HYPERPARAMETERS
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
# REWARD SHAPING
# ============================================================

SHAPING_SCALE = 1.0


# ============================================================
# DEVICE
# ============================================================

# Keep CPU explicit for this diagnostic so its behaviour is
# easy to reproduce across machines.
DEVICE = "cpu"


# ============================================================
# WAREHOUSE
# ============================================================

def create_warehouse():
    """
    Create the fixed 14 x 17 warehouse.

    Front cross aisle:
        row 0

    Rear cross aisle:
        row 13

    Picking aisle columns:
        1, 3, 5, 7, 9, 11, 13, 15

    Pick rows:
        1..12

    Depot:
        (0, 0)

    Total pick locations:
        96
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

    return (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    )


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
# ENVIRONMENT
# ============================================================

def create_environment(
    grid,
    depot,
    all_pick_locations,
):
    """
    Create WarehouseEnv using the project's fixed rewards.
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
# SET-DQN AGENT
# ============================================================

def create_agent(
    env,
):
    """
    Create a fresh permutation-invariant Set-DQN.
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
# BALANCED TRAINING SCHEDULE
# ============================================================

def build_balanced_training_schedule(
    all_pick_locations,
):
    """
    Build the complete one-pick training schedule.

    Each cycle contains all 96 pick locations exactly once,
    shuffled deterministically.

    Returns
    -------
    list[dict]
        One record per training episode.
    """

    schedule = []

    episode = 0

    for cycle_index in range(
        TRAIN_CYCLES
    ):

        cycle_number = (
            cycle_index
            + 1
        )

        locations = [
            tuple(
                location
            )
            for location in all_pick_locations
        ]

        cycle_rng = random.Random(
            TRAIN_SEED
            + 40_000
            + cycle_index
        )

        cycle_rng.shuffle(
            locations
        )

        for (
            cycle_position,
            target,
        ) in enumerate(
            locations,
            start=1,
        ):

            episode += 1

            schedule.append(
                {
                    "episode":
                        episode,

                    "cycle":
                        cycle_number,

                    "cycle_position":
                        cycle_position,

                    "order":
                        [
                            target
                        ],

                    "target":
                        target,
                }
            )

    return schedule


# ============================================================
# SCHEDULE VALIDATION
# ============================================================

def validate_balanced_training_schedule(
    schedule,
    all_pick_locations,
):
    """
    Verify exact one-pick exposure balance.
    """

    location_count = len(
        all_pick_locations
    )

    expected_episodes = (
        location_count
        * TRAIN_CYCLES
    )

    # --------------------------------------------------------
    # Overall episode count
    # --------------------------------------------------------

    if len(
        schedule
    ) != expected_episodes:

        raise AssertionError(
            f"Training schedule contains "
            f"{len(schedule)} episodes; expected "
            f"{expected_episodes}."
        )

    # --------------------------------------------------------
    # Validate each cycle
    # --------------------------------------------------------

    for cycle_number in range(
        1,
        TRAIN_CYCLES + 1,
    ):

        cycle_rows = [
            row
            for row in schedule
            if row[
                "cycle"
            ] == cycle_number
        ]

        if len(
            cycle_rows
        ) != location_count:

            raise AssertionError(
                f"Cycle {cycle_number} contains "
                f"{len(cycle_rows)} episodes instead of "
                f"{location_count}."
            )

        cycle_targets = [
            row[
                "target"
            ]
            for row in cycle_rows
        ]

        cycle_counts = Counter(
            cycle_targets
        )

        if len(
            cycle_counts
        ) != location_count:

            raise AssertionError(
                f"Cycle {cycle_number} does not contain "
                "all warehouse locations."
            )

        for location in all_pick_locations:

            if cycle_counts[
                location
            ] != 1:

                raise AssertionError(
                    f"Location {location} occurs "
                    f"{cycle_counts[location]} times in "
                    f"cycle {cycle_number}; expected 1."
                )

    # --------------------------------------------------------
    # Global exposure
    # --------------------------------------------------------

    global_counts = Counter(
        row[
            "target"
        ]
        for row in schedule
    )

    for location in all_pick_locations:

        if global_counts[
            location
        ] != TRAIN_CYCLES:

            raise AssertionError(
                f"Location {location} occurs "
                f"{global_counts[location]} times overall; "
                f"expected {TRAIN_CYCLES}."
            )


# ============================================================
# COLLECTION METRICS
# ============================================================

def get_collection_metrics(
    environment_state,
    all_pick_locations,
    order_size,
):
    """
    Calculate collection progress from raw WarehouseEnv state.
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
    Infer valid and invalid movement actions.

    Every valid movement increases travel_distance by exactly 1.
    """

    valid_actions = int(
        round(
            travel_distance
        )
    )

    invalid_actions = (
        int(
            steps
        )
        - valid_actions
    )

    invalid_actions = max(
        0,
        invalid_actions,
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
# ONE TRAINING EPISODE
# ============================================================

def train_episode(
    env,
    agent,
    order,
    grid,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Train Set-DQN for one one-pick episode.

    IMPORTANT:
        Set-DQN consumes the raw 98-dimensional environment state.

    There is no CoordinateStateEncoder.
    """

    environment_state = env.reset(
        order
    )

    if environment_state.shape != (
        98,
    ):

        raise RuntimeError(
            "Expected raw WarehouseEnv state dimension 98; "
            f"received {environment_state.shape}."
        )

    terminated = False
    truncated = False

    total_training_reward = 0.0

    total_base_reward = 0.0

    total_shaping_reward = 0.0

    losses = []

    while not (
        terminated
        or truncated
    ):

        # ----------------------------------------------------
        # Epsilon-greedy Set-DQN action
        # ----------------------------------------------------

        action = agent.select_action(
            environment_state
        )

        # ----------------------------------------------------
        # Environment transition
        # ----------------------------------------------------

        (
            next_environment_state,
            base_reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        # ----------------------------------------------------
        # Potential-based reward shaping
        # ----------------------------------------------------

        shaping_reward = (
            calculate_potential_shaping_reward(
                environment_state,
                next_environment_state,
                grid.shape,
                depot,
                all_pick_locations,
                distance_lookup,
                gamma=GAMMA,
                scale=SHAPING_SCALE,
            )
        )

        training_reward = (
            float(
                base_reward
            )
            + float(
                shaping_reward
            )
        )

        # ----------------------------------------------------
        # Replay
        #
        # terminated and truncated remain separate.
        # ----------------------------------------------------

        agent.remember(
            environment_state,
            action,
            training_reward,
            next_environment_state,
            terminated,
            truncated,
        )

        # ----------------------------------------------------
        # Gradient update
        # ----------------------------------------------------

        loss = agent.train_step()

        if loss is not None:

            if not np.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite Set-DQN loss: {loss}"
                )

            losses.append(
                float(
                    loss
                )
            )

        total_base_reward += float(
            base_reward
        )

        total_shaping_reward += float(
            shaping_reward
        )

        total_training_reward += float(
            training_reward
        )

        environment_state = (
            next_environment_state
        )

    # --------------------------------------------------------
    # Final collection metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Movement metrics
    # --------------------------------------------------------

    (
        valid_actions,
        invalid_actions,
        invalid_action_rate,
    ) = calculate_action_metrics(
        env.steps,
        env.travel_distance,
    )

    if losses:

        average_loss = float(
            np.mean(
                losses
            )
        )

    else:

        average_loss = np.nan

    return {
        "reward":
            float(
                total_training_reward
            ),

        "base_reward":
            float(
                total_base_reward
            ),

        "shaping_reward":
            float(
                total_shaping_reward
            ),

        "distance":
            float(
                env.travel_distance
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

        "epsilon":
            float(
                agent.epsilon
            ),

        "average_loss":
            average_loss,

        "training_updates":
            len(
                losses
            ),
    }


# ============================================================
# GREEDY EVALUATION OF ONE TARGET
# ============================================================

def evaluate_dqn_order(
    env,
    agent,
    order,
    optimal_distance,
    all_pick_locations,
):
    """
    Greedily evaluate one one-pick order.
    """

    environment_state = env.reset(
        order
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

        # Exact benchmark should never be longer than DQN.
        if optimality_gap < -1e-6:

            raise RuntimeError(
                "Set-DQN route appears shorter than exact "
                "optimal distance."
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

        "items_collected":
            items_collected,

        "items_remaining":
            items_remaining,

        "collection_fraction":
            collection_fraction,

        "runtime":
            float(
                runtime
            ),
    }


# ============================================================
# EXACT OPTIMAL DISTANCES
# ============================================================

def calculate_all_optima(
    grid,
    depot,
    all_pick_locations,
):
    """
    Calculate exact one-pick optimum for every warehouse target.

    Returns
    -------
    dict
        location -> optimal distance
    """

    optima = {}

    print(
        "\nCalculating exact one-pick distances "
        "for all 96 targets..."
    )

    for (
        index,
        location,
    ) in enumerate(
        all_pick_locations,
        start=1,
    ):

        order = [
            location
        ]

        optimum = exact_optimal_distance(
            grid,
            depot,
            order,
        )

        optima[
            tuple(
                location
            )
        ] = float(
            optimum
        )

        if (
            index == 1
            or
            index % 12 == 0
        ):

            print(
                f"  Calculated "
                f"{index:02d}/{len(all_pick_locations)}"
            )

    print(
        "Exact one-pick distances complete."
    )

    return optima


# ============================================================
# COMPLETE GREEDY EVALUATION
# ============================================================

def evaluate_all_locations(
    env,
    agent,
    all_pick_locations,
    optimal_distances,
    episode,
    cycle,
):
    """
    Greedily evaluate all 96 possible one-pick targets.
    """

    raw_results = []

    for (
        target_index,
        target,
    ) in enumerate(
        all_pick_locations,
        start=1,
    ):

        order = [
            target
        ]

        optimum = optimal_distances[
            tuple(
                target
            )
        ]

        evaluation = evaluate_dqn_order(
            env,
            agent,
            order,
            optimum,
            all_pick_locations,
        )

        raw_results.append(
            {
                "episode":
                    episode,

                "cycle":
                    cycle,

                "target_index":
                    target_index,

                "target":
                    str(
                        target
                    ),

                **evaluation,
            }
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    completed_results = [
        row
        for row in raw_results
        if row[
            "completed"
        ]
    ]

    completion_rate = (
        100.0
        * len(
            completed_results
        )
        / len(
            raw_results
        )
    )

    # --------------------------------------------------------
    # Collection
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Invalid actions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Route quality
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
            / len(
                completed_results
            )
        )

    else:

        mean_completed_distance = np.nan

        mean_optimality_gap = np.nan

        median_optimality_gap = np.nan

        max_optimality_gap = np.nan

        exact_optimal_count = 0

        exact_optimal_rate_completed = np.nan

    summary = {
        "episode":
            episode,

        "cycle":
            cycle,

        "completion_rate":
            completion_rate,

        "completed_orders":
            len(
                completed_results
            ),

        "total_orders":
            len(
                raw_results
            ),

        "any_pick_rate":
            any_pick_rate,

        "mean_collection_fraction":
            mean_collection_fraction,

        "mean_invalid_action_rate":
            mean_invalid_action_rate,

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
# BEST CHECKPOINT
# ============================================================

def evaluation_is_better(
    new_summary,
    best_summary,
):
    """
    Development checkpoint priority:

        1. higher completion rate
        2. higher collection fraction
        3. lower mean optimality gap
        4. lower invalid-action rate
    """

    if best_summary is None:
        return True

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    if (
        new_summary[
            "completion_rate"
        ]
        >
        best_summary[
            "completion_rate"
        ]
    ):
        return True

    if (
        new_summary[
            "completion_rate"
        ]
        <
        best_summary[
            "completion_rate"
        ]
    ):
        return False

    # --------------------------------------------------------
    # Collection
    # --------------------------------------------------------

    if (
        new_summary[
            "mean_collection_fraction"
        ]
        >
        best_summary[
            "mean_collection_fraction"
        ]
    ):
        return True

    if (
        new_summary[
            "mean_collection_fraction"
        ]
        <
        best_summary[
            "mean_collection_fraction"
        ]
    ):
        return False

    # --------------------------------------------------------
    # Optimality gap
    # --------------------------------------------------------

    new_gap = (
        new_summary[
            "mean_optimality_gap"
        ]
    )

    best_gap = (
        best_summary[
            "mean_optimality_gap"
        ]
    )

    new_gap_finite = np.isfinite(
        new_gap
    )

    best_gap_finite = np.isfinite(
        best_gap
    )

    if (
        new_gap_finite
        and
        not best_gap_finite
    ):
        return True

    if (
        best_gap_finite
        and
        not new_gap_finite
    ):
        return False

    if (
        new_gap_finite
        and
        best_gap_finite
    ):

        if new_gap < best_gap:
            return True

        if new_gap > best_gap:
            return False

    # --------------------------------------------------------
    # Invalid actions
    # --------------------------------------------------------

    return (
        new_summary[
            "mean_invalid_action_rate"
        ]
        <
        best_summary[
            "mean_invalid_action_rate"
        ]
    )


# ============================================================
# CSV SAVING
# ============================================================

def save_csv(
    rows,
    output_path,
):
    """
    Save list-of-dictionary records to CSV.
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
# CONFIG SAVING
# ============================================================

def save_experiment_config(
    output_path,
    location_count,
    total_training_episodes,
):
    """
    Save experimental settings.
    """

    config = {
        "run_name":
            RUN_NAME,

        "agent":
            "SetDQNAgent",

        "representation":
            "permutation_invariant_shared_pick_encoder_mean_pooling",

        "raw_environment_state_dimension":
            2
            + location_count,

        "coordinate_slot_encoder":
            False,

        "training_seed":
            TRAIN_SEED,

        "training_cycles":
            TRAIN_CYCLES,

        "order_size":
            ORDER_SIZE,

        "orders_per_cycle":
            location_count,

        "total_training_episodes":
            total_training_episodes,

        "exposure_per_location":
            TRAIN_CYCLES,

        "evaluation_locations":
            location_count,

        "evaluation_policy":
            "all_96_training_locations_greedy",

        "max_steps":
            MAX_STEPS,

        "max_order_size":
            MAX_ORDER_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "gamma":
            GAMMA,

        "epsilon_start":
            EPSILON_START,

        "epsilon_min":
            EPSILON_MIN,

        "epsilon_decay":
            EPSILON_DECAY,

        "batch_size":
            BATCH_SIZE,

        "buffer_capacity":
            BUFFER_CAPACITY,

        "target_update":
            TARGET_UPDATE,

        "hidden_dim":
            HIDDEN_DIM,

        "pick_embedding_dim":
            PICK_EMBEDDING_DIM,

        "shaping_scale":
            SHAPING_SCALE,

        "reward_shaping":
            "potential_MST",

        "device":
            DEVICE,
    }

    with output_path.open(
        "w"
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "================================================\n"
        "SET-DQN BALANCED ONE-PICK TRAINING\n"
        "================================================"
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

    location_count = len(
        all_pick_locations
    )

    orders_per_cycle = (
        location_count
    )

    total_training_episodes = (
        orders_per_cycle
        * TRAIN_CYCLES
    )

    evaluate_every = (
        orders_per_cycle
        * EVALUATE_EVERY_CYCLES
    )

    # ========================================================
    # ENVIRONMENT / AGENT
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
    # BASIC ARCHITECTURE CHECK
    # ========================================================

    test_state = env.reset(
        [
            all_pick_locations[
                0
            ]
        ]
    )

    q_values = agent.get_q_values(
        test_state
    )

    if test_state.shape != (
        98,
    ):

        raise RuntimeError(
            f"Expected raw environment state (98,), "
            f"received {test_state.shape}."
        )

    if q_values.shape != (
        4,
    ):

        raise RuntimeError(
            f"Expected four Q-values, "
            f"received {q_values.shape}."
        )

    print(
        f"Run name:               {RUN_NAME}\n"
        f"Agent:                  SetDQN\n"
        f"Training cycles:        {TRAIN_CYCLES}\n"
        f"Locations:              {location_count}\n"
        f"Orders per cycle:       {orders_per_cycle}\n"
        f"Training episodes:      {total_training_episodes}\n"
        f"Order size:             {ORDER_SIZE}\n"
        f"Exposure per location:  {TRAIN_CYCLES}\n"
        f"Greedy eval locations:  {location_count}\n"
        f"Evaluate every:         {evaluate_every} episodes\n"
        f"Max episode steps:      {MAX_STEPS}\n"
        f"Training seed:          {TRAIN_SEED}\n"
        f"Epsilon decay:          {EPSILON_DECAY}\n"
        f"Shaping scale:          {SHAPING_SCALE}\n"
        f"Raw state dimension:    {test_state.shape[0]}\n"
        f"Pick embedding dim:     {PICK_EMBEDDING_DIM}\n"
        f"Device:                 {DEVICE}\n"
    )

    # ========================================================
    # TRAINING SCHEDULE
    # ========================================================

    print(
        "Building balanced one-pick training schedule..."
    )

    training_schedule = (
        build_balanced_training_schedule(
            all_pick_locations
        )
    )

    validate_balanced_training_schedule(
        training_schedule,
        all_pick_locations,
    )

    print(
        "Balanced schedule verified:"
    )

    print(
        f"  {orders_per_cycle} episodes per cycle"
    )

    print(
        f"  {TRAIN_CYCLES} cycles"
    )

    print(
        f"  {total_training_episodes} episodes"
    )

    print(
        f"  {TRAIN_CYCLES} exposures per target"
    )

    # ========================================================
    # DISTANCE LOOKUP
    # ========================================================

    print(
        "\nPre-computing shortest-path distances..."
    )

    distance_lookup = build_distance_lookup(
        grid,
        depot,
        all_pick_locations,
    )

    print(
        "Shortest-path lookup complete."
    )

    # ========================================================
    # EXACT OPTIMAL DISTANCES
    # ========================================================

    optimal_distances = calculate_all_optima(
        grid,
        depot,
        all_pick_locations,
    )

    print(
        f"\nMinimum exact distance: "
        f"{min(optimal_distances.values()):.0f}"
    )

    print(
        f"Maximum exact distance: "
        f"{max(optimal_distances.values()):.0f}"
    )

    # ========================================================
    # OUTPUT DIRECTORY
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

    schedule_path = (
        results_directory
        / "training_schedule.csv"
    )

    evaluation_targets_path = (
        results_directory
        / "evaluation_targets.csv"
    )

    training_history_path = (
        results_directory
        / "set_dqn_training_history.csv"
    )

    evaluation_summary_path = (
        results_directory
        / "set_dqn_evaluation_summary.csv"
    )

    evaluation_raw_path = (
        results_directory
        / "set_dqn_evaluation_raw.csv"
    )

    best_model_path = (
        results_directory
        / "set_dqn_best_model.pt"
    )

    final_model_path = (
        results_directory
        / "set_dqn_final_model.pt"
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    save_experiment_config(
        config_path,
        location_count,
        total_training_episodes,
    )

    # ========================================================
    # SAVE SCHEDULE
    # ========================================================

    save_csv(
        [
            {
                "episode":
                    row[
                        "episode"
                    ],

                "cycle":
                    row[
                        "cycle"
                    ],

                "cycle_position":
                    row[
                        "cycle_position"
                    ],

                "target":
                    str(
                        row[
                            "target"
                        ]
                    ),
            }
            for row in training_schedule
        ],
        schedule_path,
    )

    # ========================================================
    # SAVE EVALUATION TARGETS
    # ========================================================

    save_csv(
        [
            {
                "target_index":
                    index,

                "target":
                    str(
                        target
                    ),

                "optimal_distance":
                    optimal_distances[
                        tuple(
                            target
                        )
                    ],
            }
            for (
                index,
                target,
            ) in enumerate(
                all_pick_locations,
                start=1,
            )
        ],
        evaluation_targets_path,
    )

    # ========================================================
    # HISTORIES
    # ========================================================

    training_history = []

    evaluation_summaries = []

    evaluation_raw_results = []

    rolling_completion = deque(
        maxlen=ROLLING_WINDOW
    )

    rolling_collection = deque(
        maxlen=ROLLING_WINDOW
    )

    rolling_invalid_rate = deque(
        maxlen=ROLLING_WINDOW
    )

    rolling_reward = deque(
        maxlen=ROLLING_WINDOW
    )

    best_evaluation_summary = None

    # ========================================================
    # UNTRAINED EVALUATION
    # ========================================================

    print(
        "\nEvaluating untrained Set-DQN on all 96 targets..."
    )

    (
        initial_summary,
        initial_raw,
    ) = evaluate_all_locations(
        env,
        agent,
        all_pick_locations,
        optimal_distances,
        episode=0,
        cycle=0,
    )

    evaluation_summaries.append(
        initial_summary
    )

    evaluation_raw_results.extend(
        initial_raw
    )

    best_evaluation_summary = (
        initial_summary.copy()
    )

    agent.save(
        best_model_path
    )

    print(
        f"Episode 0"
        f" | completion "
        f"{initial_summary['completion_rate']:.1f}%"
        f" | any-pick "
        f"{initial_summary['any_pick_rate']:.1f}%"
        f" | collection "
        f"{initial_summary['mean_collection_fraction']:.3f}"
        f" | invalid "
        f"{initial_summary['mean_invalid_action_rate']:.1f}%"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    training_start = (
        time.perf_counter()
    )

    for schedule_row in training_schedule:

        episode = schedule_row[
            "episode"
        ]

        cycle = schedule_row[
            "cycle"
        ]

        cycle_position = schedule_row[
            "cycle_position"
        ]

        order = schedule_row[
            "order"
        ]

        target = schedule_row[
            "target"
        ]

        metrics = train_episode(
            env,
            agent,
            order,
            grid,
            depot,
            all_pick_locations,
            distance_lookup,
        )

        # ----------------------------------------------------
        # Rolling metrics
        # ----------------------------------------------------

        rolling_completion.append(
            int(
                metrics[
                    "completed"
                ]
            )
        )

        rolling_collection.append(
            metrics[
                "collection_fraction"
            ]
        )

        rolling_invalid_rate.append(
            metrics[
                "invalid_action_rate"
            ]
        )

        rolling_reward.append(
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
                rolling_collection
            )
        )

        rolling_mean_invalid = float(
            np.mean(
                rolling_invalid_rate
            )
        )

        rolling_mean_reward = float(
            np.mean(
                rolling_reward
            )
        )

        # ----------------------------------------------------
        # Training history
        # ----------------------------------------------------

        training_history.append(
            {
                "episode":
                    episode,

                "cycle":
                    cycle,

                "cycle_position":
                    cycle_position,

                "target":
                    str(
                        target
                    ),

                **metrics,

                "rolling_96_completion_rate":
                    rolling_completion_rate,

                "rolling_96_collection_fraction":
                    rolling_mean_collection,

                "rolling_96_invalid_action_rate":
                    rolling_mean_invalid,

                "rolling_96_reward":
                    rolling_mean_reward,
            }
        )

        # ----------------------------------------------------
        # Console progress
        # ----------------------------------------------------

        if (
            episode == 1
            or
            episode % PRINT_EVERY
            == 0
        ):

            if np.isfinite(
                metrics[
                    "average_loss"
                ]
            ):

                loss_text = (
                    f"{metrics['average_loss']:.4f}"
                )

            else:

                loss_text = "N/A"

            print(
                f"Episode "
                f"{episode:04d}/{total_training_episodes}"
                f" | cycle "
                f"{cycle:02d}/{TRAIN_CYCLES}"
                f" | pos "
                f"{cycle_position:02d}/{orders_per_cycle}"
                f" | target "
                f"{target}"
                f" | collect "
                f"{metrics['items_collected']}/1"
                f" | complete "
                f"{int(metrics['completed'])}"
                f" | dist "
                f"{metrics['distance']:3.0f}"
                f" | invalid "
                f"{metrics['invalid_action_rate']:5.1f}%"
                f" | roll complete "
                f"{rolling_completion_rate:5.1f}%"
                f" | eps "
                f"{metrics['epsilon']:.3f}"
                f" | loss "
                f"{loss_text}"
            )

        # ====================================================
        # GREEDY ALL-LOCATION EVALUATION
        # ====================================================

        if (
            episode
            % evaluate_every
            == 0
        ):

            (
                evaluation_summary,
                evaluation_raw,
            ) = evaluate_all_locations(
                env,
                agent,
                all_pick_locations,
                optimal_distances,
                episode=episode,
                cycle=cycle,
            )

            evaluation_summaries.append(
                evaluation_summary
            )

            evaluation_raw_results.extend(
                evaluation_raw
            )

            mean_gap = (
                evaluation_summary[
                    "mean_optimality_gap"
                ]
            )

            median_gap = (
                evaluation_summary[
                    "median_optimality_gap"
                ]
            )

            if np.isfinite(
                mean_gap
            ):

                mean_gap_text = (
                    f"{mean_gap:.2f}%"
                )

            else:

                mean_gap_text = "N/A"

            if np.isfinite(
                median_gap
            ):

                median_gap_text = (
                    f"{median_gap:.2f}%"
                )

            else:

                median_gap_text = "N/A"

            print(
                "\n"
                "------------------------------------------------"
            )

            print(
                f"SET-DQN ONE-PICK EVALUATION "
                f"AFTER CYCLE {cycle}/{TRAIN_CYCLES}"
            )

            print(
                "------------------------------------------------"
            )

            print(
                f"Episode:                  "
                f"{episode}"
            )

            print(
                f"Completion rate:          "
                f"{evaluation_summary['completion_rate']:.1f}%"
            )

            print(
                f"Completed targets:        "
                f"{evaluation_summary['completed_orders']}"
                f"/{evaluation_summary['total_orders']}"
            )

            print(
                f"Any-pick rate:            "
                f"{evaluation_summary['any_pick_rate']:.1f}%"
            )

            print(
                f"Collection fraction:      "
                f"{evaluation_summary['mean_collection_fraction']:.3f}"
            )

            print(
                f"Mean invalid-action rate: "
                f"{evaluation_summary['mean_invalid_action_rate']:.1f}%"
            )

            print(
                f"Mean completed distance:  "
                f"{evaluation_summary['mean_completed_distance']}"
            )

            print(
                f"Mean optimality gap:      "
                f"{mean_gap_text}"
            )

            print(
                f"Median optimality gap:    "
                f"{median_gap_text}"
            )

            print(
                f"Exact-optimal completions:"
                f" {evaluation_summary['exact_optimal_count']}"
                f"/{evaluation_summary['completed_orders']}"
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if evaluation_is_better(
                evaluation_summary,
                best_evaluation_summary,
            ):

                best_evaluation_summary = (
                    evaluation_summary.copy()
                )

                agent.save(
                    best_model_path
                )

                print(
                    "New best one-pick Set-DQN model saved."
                )

            print()

            # ------------------------------------------------
            # Persist intermediate results
            # ------------------------------------------------

            save_csv(
                training_history,
                training_history_path,
            )

            save_csv(
                evaluation_summaries,
                evaluation_summary_path,
            )

            save_csv(
                evaluation_raw_results,
                evaluation_raw_path,
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    training_time = (
        time.perf_counter()
        - training_start
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    agent.save(
        final_model_path
    )

    # ========================================================
    # FINAL RESULT SAVING
    # ========================================================

    save_csv(
        training_history,
        training_history_path,
    )

    save_csv(
        evaluation_summaries,
        evaluation_summary_path,
    )

    save_csv(
        evaluation_raw_results,
        evaluation_raw_path,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        "================================================\n"
        "SET-DQN BALANCED ONE-PICK TRAINING COMPLETE\n"
        "================================================"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Training episodes: "
        f"{total_training_episodes}"
    )

    print(
        f"Exposure per location: "
        f"{TRAIN_CYCLES}"
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
        f"Final rolling collection: "
        f"{np.mean(rolling_collection):.3f}"
    )

    print(
        f"Final rolling invalid rate: "
        f"{np.mean(rolling_invalid_rate):.1f}%"
    )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if best_evaluation_summary is not None:

        print(
            "\nBest one-pick Set-DQN checkpoint:"
        )

        print(
            f"  Episode: "
            f"{best_evaluation_summary['episode']}"
        )

        print(
            f"  Cycle: "
            f"{best_evaluation_summary['cycle']}"
        )

        print(
            f"  Completion: "
            f"{best_evaluation_summary['completion_rate']:.1f}%"
        )

        print(
            f"  Any-pick: "
            f"{best_evaluation_summary['any_pick_rate']:.1f}%"
        )

        print(
            f"  Collection fraction: "
            f"{best_evaluation_summary['mean_collection_fraction']:.3f}"
        )

        print(
            f"  Invalid-action rate: "
            f"{best_evaluation_summary['mean_invalid_action_rate']:.1f}%"
        )

        best_gap = (
            best_evaluation_summary[
                "mean_optimality_gap"
            ]
        )

        if np.isfinite(
            best_gap
        ):

            print(
                f"  Mean optimality gap: "
                f"{best_gap:.2f}%"
            )

        print(
            f"  Exact-optimal completions: "
            f"{best_evaluation_summary['exact_optimal_count']}"
            f"/{best_evaluation_summary['completed_orders']}"
        )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Config:\n"
        f"    {config_path}"
    )

    print(
        f"  Training schedule:\n"
        f"    {schedule_path}"
    )

    print(
        f"  Evaluation targets:\n"
        f"    {evaluation_targets_path}"
    )

    print(
        f"  Training history:\n"
        f"    {training_history_path}"
    )

    print(
        f"  Evaluation summary:\n"
        f"    {evaluation_summary_path}"
    )

    print(
        f"  Evaluation raw:\n"
        f"    {evaluation_raw_path}"
    )

    print(
        f"  Best model:\n"
        f"    {best_model_path}"
    )

    print(
        f"  Final model:\n"
        f"    {final_model_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
