"""
Balanced two-pick DQN training using coordinate-based state encoding.

Purpose
-------
Progress from the successful balanced one-pick coordinate experiment
to orders containing two pick locations.

The DQN uses the 62-dimensional coordinate state:

    2 picker coordinates
    +
    20 pick slots x 3 values

where each pick slot contains:

    normalised row
    normalised column
    active flag

Training schedule
-----------------
There are 96 possible pick locations.

During each cycle:
    - every location appears exactly once
    - locations are paired into 48 two-pick orders
    - no development-probe pair is allowed to appear in training

Training:
    48 orders per cycle
    x 60 cycles
    = 2880 episodes

Each individual location therefore appears 60 times during training.

Development probes
------------------
40 fixed two-pick combinations are used.

The exact two-location combinations are excluded from every
training cycle.

However, the individual locations contained in those probes still
appear in other training combinations.

Therefore this experiment tests generalisation to unseen
COMBINATIONS of familiar warehouse locations.

The development probes are NOT the final project test set.

Reward shaping
--------------
Uses the multi-pick potential-based shaping implementation from:

    utils/reward_shaping.py

The shaping potential uses:
    - current-to-remaining-set distance
    - MST connection cost
    - remaining-set-to-depot distance

It does not use the exact routing optimiser.

Exact optimal distances are used only for evaluation.
"""

from utils.state_encoding import CoordinateStateEncoder
from utils.reward_shaping import (
    build_distance_lookup,
    calculate_potential_shaping_reward,
    get_remaining_pick_locations,
)
from utils.order_generation import generate_order
from exact.ratliff_rosenthal import exact_optimal_distance
from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent
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

RUN_NAME = "balanced_two_pick_coordinates_60_cycles"

TRAIN_SEED = 42

TRAIN_CYCLES = 60

ORDER_SIZE = 2

MAX_ORDER_SIZE = 20

MAX_STEPS = 150


# ============================================================
# DEVELOPMENT PROBES
# ============================================================

PROBE_ORDERS = 40

PROBE_SEED_START = 70_000

EVALUATE_EVERY_CYCLES = 1


# ============================================================
# TRAINING METRICS
# ============================================================

# One full two-pick cycle:
#
# 96 locations / 2 picks = 48 episodes
ROLLING_WINDOW = 48

# Print approximately twice per cycle.
PRINT_EVERY = 24


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


# ============================================================
# REWARD SHAPING
# ============================================================

SHAPING_SCALE = 1.0


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
        8 aisles x 12 rows = 96
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
    Set Python, NumPy and PyTorch seeds.
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
    Create the project's WarehouseEnv.
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
# COORDINATE STATE ENCODER
# ============================================================

def create_state_encoder(
    grid,
    all_pick_locations,
):
    """
    Create the 62-dimensional coordinate encoder.
    """

    return CoordinateStateEncoder(
        all_pick_locations=all_pick_locations,
        grid_shape=grid.shape,
        max_order_size=MAX_ORDER_SIZE,
    )


# ============================================================
# DQN AGENT
# ============================================================

def create_agent(
    env,
    encoder,
):
    """
    Create the DQN configured for the coordinate state.
    """

    return DQNAgent(
        env=env,

        input_dim=encoder.output_dim,

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
# ORDER KEY
# ============================================================

def order_key(
    order,
):
    """
    Create an order-independent representation of a
    two-pick combination.

    For example:

        [(2,1), (8,7)]

    and:

        [(8,7), (2,1)]

    produce the same key.
    """

    return tuple(
        sorted(
            tuple(
                location
            )
            for location in order
        )
    )


# ============================================================
# DEVELOPMENT PROBES
# ============================================================

def create_probe_orders(
    all_pick_locations,
    aisle_columns,
):
    """
    Generate 40 deterministic unique two-pick development
    probe combinations.

    These exact combinations will later be prohibited from
    appearing in the training schedule.
    """

    probe_orders = []

    seen_keys = set()

    seed = PROBE_SEED_START

    max_attempts = 100_000

    attempts = 0

    while len(
        probe_orders
    ) < PROBE_ORDERS:

        attempts += 1

        if attempts > max_attempts:

            raise RuntimeError(
                "Unable to generate enough unique "
                "two-pick probe orders."
            )

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=ORDER_SIZE,
            distribution="uniform",
            seed=seed,
            aisle_columns=aisle_columns,
        )

        key = order_key(
            order
        )

        if key not in seen_keys:

            probe_orders.append(
                list(
                    key
                )
            )

            seen_keys.add(
                key
            )

        seed += 1

    return probe_orders


# ============================================================
# BALANCED TWO-PICK TRAINING SCHEDULE
# ============================================================

def build_balanced_training_schedule(
    all_pick_locations,
    forbidden_pair_keys,
):
    """
    Build balanced two-pick training cycles.

    During every cycle:

        - all 96 locations appear exactly once
        - they are grouped into 48 two-pick orders
        - no order matches a development probe combination

    Parameters
    ----------
    all_pick_locations
        All 96 fixed pick coordinates.

    forbidden_pair_keys
        Set of two-pick development probe combinations.

    Returns
    -------
    list[dict]
        Complete training schedule.
    """

    if len(
        all_pick_locations
    ) % 2 != 0:

        raise ValueError(
            "The number of pick locations must be even "
            "for two-pick pairing."
        )

    orders_per_cycle = (
        len(
            all_pick_locations
        )
        // ORDER_SIZE
    )

    schedule = []

    episode = 0

    for cycle_index in range(
        TRAIN_CYCLES
    ):

        cycle_number = (
            cycle_index
            + 1
        )

        # Use a separate deterministic RNG for each cycle.
        cycle_rng = random.Random(
            TRAIN_SEED
            + 10_000
            + cycle_index
        )

        valid_pairs = None

        # A random pairing almost always avoids the relatively
        # small forbidden probe set, but retry deterministically
        # if necessary.
        for attempt in range(
            10_000
        ):

            locations = list(
                all_pick_locations
            )

            cycle_rng.shuffle(
                locations
            )

            candidate_pairs = []

            contains_forbidden_pair = False

            for start_index in range(
                0,
                len(
                    locations
                ),
                ORDER_SIZE,
            ):

                order = [
                    locations[
                        start_index
                    ],
                    locations[
                        start_index
                        + 1
                    ],
                ]

                key = order_key(
                    order
                )

                if key in forbidden_pair_keys:

                    contains_forbidden_pair = True
                    break

                candidate_pairs.append(
                    list(
                        key
                    )
                )

            if not contains_forbidden_pair:

                valid_pairs = (
                    candidate_pairs
                )

                break

        if valid_pairs is None:

            raise RuntimeError(
                f"Unable to construct cycle "
                f"{cycle_number} without development "
                "probe combinations."
            )

        if len(
            valid_pairs
        ) != orders_per_cycle:

            raise AssertionError(
                "Incorrect number of two-pick orders "
                "in training cycle."
            )

        for (
            cycle_position,
            order,
        ) in enumerate(
            valid_pairs,
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
                        order,

                    "order_key":
                        order_key(
                            order
                        ),
                }
            )

    return schedule


# ============================================================
# TRAINING-SCHEDULE VALIDATION
# ============================================================

def validate_balanced_training_schedule(
    schedule,
    all_pick_locations,
    forbidden_pair_keys,
):
    """
    Verify all balancing and probe-exclusion requirements.
    """

    location_count = len(
        all_pick_locations
    )

    orders_per_cycle = (
        location_count
        // ORDER_SIZE
    )

    expected_episodes = (
        orders_per_cycle
        * TRAIN_CYCLES
    )

    # --------------------------------------------------------
    # Total episode count
    # --------------------------------------------------------

    if len(
        schedule
    ) != expected_episodes:

        raise AssertionError(
            f"Training schedule has "
            f"{len(schedule)} episodes; expected "
            f"{expected_episodes}."
        )

    # --------------------------------------------------------
    # No development combination in training
    # --------------------------------------------------------

    training_pair_keys = {
        row[
            "order_key"
        ]
        for row in schedule
    }

    overlap = (
        training_pair_keys
        & forbidden_pair_keys
    )

    if overlap:

        raise AssertionError(
            "Development probe combinations were found "
            f"in training: {sorted(overlap)}"
        )

    # --------------------------------------------------------
    # Each cycle contains each location exactly once
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
        ) != orders_per_cycle:

            raise AssertionError(
                f"Cycle {cycle_number} contains "
                f"{len(cycle_rows)} orders instead of "
                f"{orders_per_cycle}."
            )

        cycle_locations = []

        for row in cycle_rows:

            if len(
                row[
                    "order"
                ]
            ) != ORDER_SIZE:

                raise AssertionError(
                    "A training order does not contain "
                    "exactly two locations."
                )

            if len(
                set(
                    row[
                        "order"
                    ]
                )
            ) != ORDER_SIZE:

                raise AssertionError(
                    "A training order contains a "
                    "duplicate location."
                )

            cycle_locations.extend(
                row[
                    "order"
                ]
            )

        cycle_counts = Counter(
            cycle_locations
        )

        if len(
            cycle_counts
        ) != location_count:

            raise AssertionError(
                f"Cycle {cycle_number} does not contain "
                "all 96 locations."
            )

        for location in all_pick_locations:

            if cycle_counts[
                location
            ] != 1:

                raise AssertionError(
                    f"Location {location} occurs "
                    f"{cycle_counts[location]} times in "
                    f"cycle {cycle_number}."
                )

    # --------------------------------------------------------
    # Global exposure
    # --------------------------------------------------------

    all_training_locations = [
        location
        for row in schedule
        for location in row[
            "order"
        ]
    ]

    global_counts = Counter(
        all_training_locations
    )

    for location in all_pick_locations:

        if global_counts[
            location
        ] != TRAIN_CYCLES:

            raise AssertionError(
                f"Location {location} occurs "
                f"{global_counts[location]} times overall "
                f"instead of {TRAIN_CYCLES}."
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
    Calculate collection progress from the raw environment
    state.
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
    A valid action increases travel distance by exactly 1.

    Therefore:

        invalid_actions =
            steps - travel_distance
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
# TRAINING EPISODE
# ============================================================

def train_episode(
    env,
    encoder,
    agent,
    order,
    grid,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Train the DQN for one two-pick episode.

    Raw WarehouseEnv state:
        98 dimensions

    DQN state:
        62 coordinate features

    Reward shaping uses the raw environment state.

    Replay memory stores the encoded DQN states.
    """

    # --------------------------------------------------------
    # Reset environment and encoder
    # --------------------------------------------------------

    environment_state = (
        env.reset(
            order
        )
    )

    encoder.reset(
        order
    )

    dqn_state = encoder.encode(
        environment_state
    )

    if dqn_state.shape != (
        encoder.output_dim,
    ):

        raise RuntimeError(
            "Encoded DQN state has incorrect dimension."
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
        # Epsilon-greedy action
        # ----------------------------------------------------

        action = agent.select_action(
            dqn_state
        )

        # ----------------------------------------------------
        # Raw environment transition
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
        # Multi-pick potential shaping
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
            + shaping_reward
        )

        # ----------------------------------------------------
        # Coordinate next state
        # ----------------------------------------------------

        next_dqn_state = (
            encoder.encode(
                next_environment_state
            )
        )

        # ----------------------------------------------------
        # Replay
        #
        # DQNAgent distinguishes termination from truncation.
        # ----------------------------------------------------

        agent.remember(
            dqn_state,
            action,
            training_reward,
            next_dqn_state,
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
                    f"Non-finite DQN loss detected: {loss}"
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

        dqn_state = (
            next_dqn_state
        )

    # --------------------------------------------------------
    # Final episode metrics
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
# GREEDY ORDER EVALUATION
# ============================================================

def evaluate_dqn_order(
    env,
    encoder,
    agent,
    order,
    optimal_distance,
    all_pick_locations,
):
    """
    Evaluate one two-pick order greedily.

    No replay or optimisation occurs.
    """

    environment_state = (
        env.reset(
            order
        )
    )

    encoder.reset(
        order
    )

    dqn_state = encoder.encode(
        environment_state
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
            dqn_state,
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

        next_dqn_state = encoder.encode(
            next_environment_state
        )

        environment_state = (
            next_environment_state
        )

        dqn_state = (
            next_dqn_state
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

        # A negative gap would indicate a disagreement between
        # the exact benchmark and environment distance.
        if optimality_gap < -1e-6:

            raise RuntimeError(
                "DQN route appears shorter than the exact "
                "optimum. Check benchmark consistency."
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
# EXACT PROBE OPTIMA
# ============================================================

def calculate_probe_optima(
    grid,
    depot,
    probe_orders,
):
    """
    Calculate exact optimal travel distance for every
    two-pick development probe.
    """

    optima = []

    print(
        "\nCalculating exact probe optimal distances..."
    )

    for (
        index,
        order,
    ) in enumerate(
        probe_orders,
        start=1,
    ):

        optimum = (
            exact_optimal_distance(
                grid,
                depot,
                order,
            )
        )

        optima.append(
            float(
                optimum
            )
        )

        print(
            f"  Probe "
            f"{index:02d}/{len(probe_orders)}"
            f" -> {order}"
            f" -> optimum {optimum:.0f}"
        )

    return optima


# ============================================================
# PROBE-SET EVALUATION
# ============================================================

def evaluate_probe_set(
    env,
    encoder,
    agent,
    probe_orders,
    probe_optima,
    all_pick_locations,
    episode,
    cycle,
):
    """
    Evaluate all fixed unseen-combination development probes.
    """

    raw_results = []

    for (
        probe_index,
        (
            order,
            optimum,
        ),
    ) in enumerate(
        zip(
            probe_orders,
            probe_optima,
        ),
        start=1,
    ):

        evaluation = (
            evaluate_dqn_order(
                env,
                encoder,
                agent,
                order,
                optimum,
                all_pick_locations,
            )
        )

        raw_results.append(
            {
                "episode":
                    episode,

                "cycle":
                    cycle,

                "probe_index":
                    probe_index,

                "order":
                    str(
                        order
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
    # Collection progress
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
    # Completed-route quality
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

        mean_optimal_distance = float(
            np.mean(
                [
                    row[
                        "optimal_distance"
                    ]
                    for row in completed_results
                ]
            )
        )

        completed_gaps = [
            row[
                "optimality_gap"
            ]
            for row in completed_results
        ]

        mean_optimality_gap = float(
            np.mean(
                completed_gaps
            )
        )

        max_optimality_gap = float(
            np.max(
                completed_gaps
            )
        )

    else:

        mean_completed_distance = np.nan

        mean_optimal_distance = np.nan

        mean_optimality_gap = np.nan

        max_optimality_gap = np.nan

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

        "mean_completed_distance":
            mean_completed_distance,

        "mean_optimal_distance":
            mean_optimal_distance,

        "mean_optimality_gap":
            mean_optimality_gap,

        "max_optimality_gap":
            max_optimality_gap,
    }

    return (
        summary,
        raw_results,
    )


# ============================================================
# BEST CHECKPOINT SELECTION
# ============================================================

def probe_score_is_better(
    new_summary,
    best_summary,
):
    """
    Development checkpoint selection priority:

        1. higher completion rate
        2. higher mean collection fraction
        3. lower mean optimality gap
        4. lower invalid-action rate
    """

    if best_summary is None:
        return True

    # --------------------------------------------------------
    # 1. Completion
    # --------------------------------------------------------

    new_completion = (
        new_summary[
            "completion_rate"
        ]
    )

    best_completion = (
        best_summary[
            "completion_rate"
        ]
    )

    if new_completion > best_completion:
        return True

    if new_completion < best_completion:
        return False

    # --------------------------------------------------------
    # 2. Collection
    # --------------------------------------------------------

    new_collection = (
        new_summary[
            "mean_collection_fraction"
        ]
    )

    best_collection = (
        best_summary[
            "mean_collection_fraction"
        ]
    )

    if new_collection > best_collection:
        return True

    if new_collection < best_collection:
        return False

    # --------------------------------------------------------
    # 3. Optimality gap
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
    # 4. Invalid actions
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
# CONFIG SAVING
# ============================================================

def save_experiment_config(
    output_path,
    location_count,
    orders_per_cycle,
    total_training_episodes,
    encoder_dim,
):
    """
    Save key experimental settings for reproducibility.
    """

    config = {
        "run_name":
            RUN_NAME,

        "training_seed":
            TRAIN_SEED,

        "training_cycles":
            TRAIN_CYCLES,

        "order_size":
            ORDER_SIZE,

        "location_count":
            location_count,

        "orders_per_cycle":
            orders_per_cycle,

        "total_training_episodes":
            total_training_episodes,

        "location_exposures":
            TRAIN_CYCLES,

        "probe_orders":
            PROBE_ORDERS,

        "probe_seed_start":
            PROBE_SEED_START,

        "max_steps":
            MAX_STEPS,

        "coordinate_state_dimension":
            encoder_dim,

        "max_encoded_order_size":
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

        "shaping_scale":
            SHAPING_SCALE,

        "reward_shaping":
            "potential_MST",

        "probe_policy":
            "unseen_exact_two_pick_combinations",
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
        "DQN BALANCED TWO-PICK COORDINATE TRAINING\n"
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
        // ORDER_SIZE
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
    # ENVIRONMENT + ENCODER + AGENT
    # ========================================================

    env = create_environment(
        grid,
        depot,
        all_pick_locations,
    )

    encoder = create_state_encoder(
        grid,
        all_pick_locations,
    )

    agent = create_agent(
        env,
        encoder,
    )

    if encoder.output_dim != 62:

        raise RuntimeError(
            f"Expected encoder dimension 62, "
            f"received {encoder.output_dim}."
        )

    if agent.input_dim != 62:

        raise RuntimeError(
            f"Expected DQN input dimension 62, "
            f"received {agent.input_dim}."
        )

    print(
        f"Run name:                {RUN_NAME}\n"
        f"Training cycles:         {TRAIN_CYCLES}\n"
        f"Locations:               {location_count}\n"
        f"Orders per cycle:        {orders_per_cycle}\n"
        f"Training episodes:       {total_training_episodes}\n"
        f"Order size:              {ORDER_SIZE}\n"
        f"Exposure per location:   {TRAIN_CYCLES}\n"
        f"Probe combinations:      {PROBE_ORDERS}\n"
        f"Evaluate every:          {evaluate_every} episodes\n"
        f"Max episode steps:       {MAX_STEPS}\n"
        f"Training seed:           {TRAIN_SEED}\n"
        f"Epsilon decay:           {EPSILON_DECAY}\n"
        f"Shaping scale:           {SHAPING_SCALE}\n"
        f"Raw env state dim:       {2 + location_count}\n"
        f"Encoded DQN state dim:   {encoder.output_dim}\n"
    )

    # ========================================================
    # DEVELOPMENT PROBES
    # ========================================================

    print(
        "Creating unseen-combination development probes..."
    )

    probe_orders = create_probe_orders(
        all_pick_locations,
        aisle_columns,
    )

    probe_pair_keys = {
        order_key(
            order
        )
        for order in probe_orders
    }

    if len(
        probe_pair_keys
    ) != PROBE_ORDERS:

        raise AssertionError(
            "Development probe set contains duplicates."
        )

    print(
        f"Created {len(probe_orders)} unique "
        "two-pick probe combinations."
    )

    # ========================================================
    # TRAINING SCHEDULE
    # ========================================================

    print(
        "\nBuilding balanced training schedule..."
    )

    training_schedule = (
        build_balanced_training_schedule(
            all_pick_locations,
            probe_pair_keys,
        )
    )

    validate_balanced_training_schedule(
        training_schedule,
        all_pick_locations,
        probe_pair_keys,
    )

    print(
        "Balanced schedule verified:"
    )

    print(
        f"  {orders_per_cycle} orders per cycle"
    )

    print(
        f"  {TRAIN_CYCLES} cycles"
    )

    print(
        f"  {total_training_episodes} total episodes"
    )

    print(
        f"  {TRAIN_CYCLES} exposures per location"
    )

    print(
        "  0 development-probe combinations in training"
    )

    # ========================================================
    # SHORTEST-PATH LOOKUP
    # ========================================================

    print(
        "\nPre-computing shortest-path distances..."
    )

    distance_lookup = (
        build_distance_lookup(
            grid,
            depot,
            all_pick_locations,
        )
    )

    print(
        "Shortest-path lookup complete."
    )

    # ========================================================
    # EXACT DEVELOPMENT OPTIMA
    # ========================================================

    probe_optima = (
        calculate_probe_optima(
            grid,
            depot,
            probe_orders,
        )
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

    probe_definition_path = (
        results_directory
        / "probe_orders.csv"
    )

    training_history_path = (
        results_directory
        / "dqn_training_history.csv"
    )

    probe_summary_path = (
        results_directory
        / "dqn_probe_summary.csv"
    )

    probe_raw_path = (
        results_directory
        / "dqn_probe_raw.csv"
    )

    best_model_path = (
        results_directory
        / "dqn_best_probe_model.pt"
    )

    final_model_path = (
        results_directory
        / "dqn_final_model.pt"
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    save_experiment_config(
        config_path,
        location_count,
        orders_per_cycle,
        total_training_episodes,
        encoder.output_dim,
    )

    # ========================================================
    # SAVE TRAINING SCHEDULE
    # ========================================================

    schedule_rows = [
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

            "order":
                str(
                    row[
                        "order"
                    ]
                ),
        }
        for row in training_schedule
    ]

    save_csv(
        schedule_rows,
        schedule_path,
    )

    # ========================================================
    # SAVE PROBE DEFINITIONS
    # ========================================================

    probe_definition_rows = [
        {
            "probe_index":
                index,

            "order":
                str(
                    order
                ),

            "optimal_distance":
                optimum,
        }
        for (
            index,
            (
                order,
                optimum,
            ),
        ) in enumerate(
            zip(
                probe_orders,
                probe_optima,
            ),
            start=1,
        )
    ]

    save_csv(
        probe_definition_rows,
        probe_definition_path,
    )

    # ========================================================
    # RESULT HISTORIES
    # ========================================================

    training_history = []

    probe_summaries = []

    probe_raw_results = []

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

    best_probe_summary = None

    # ========================================================
    # UNTRAINED DEVELOPMENT EVALUATION
    # ========================================================

    print(
        "\nEvaluating untrained two-pick policy..."
    )

    (
        initial_summary,
        initial_raw,
    ) = evaluate_probe_set(
        env,
        encoder,
        agent,
        probe_orders,
        probe_optima,
        all_pick_locations,
        episode=0,
        cycle=0,
    )

    probe_summaries.append(
        initial_summary
    )

    probe_raw_results.extend(
        initial_raw
    )

    best_probe_summary = (
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

        episode = (
            schedule_row[
                "episode"
            ]
        )

        cycle = (
            schedule_row[
                "cycle"
            ]
        )

        cycle_position = (
            schedule_row[
                "cycle_position"
            ]
        )

        order = (
            schedule_row[
                "order"
            ]
        )

        metrics = train_episode(
            env,
            encoder,
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

                "order":
                    str(
                        order
                    ),

                "reward":
                    metrics[
                        "reward"
                    ],

                "base_reward":
                    metrics[
                        "base_reward"
                    ],

                "shaping_reward":
                    metrics[
                        "shaping_reward"
                    ],

                "distance":
                    metrics[
                        "distance"
                    ],

                "steps":
                    metrics[
                        "steps"
                    ],

                "valid_actions":
                    metrics[
                        "valid_actions"
                    ],

                "invalid_actions":
                    metrics[
                        "invalid_actions"
                    ],

                "invalid_action_rate":
                    metrics[
                        "invalid_action_rate"
                    ],

                "completed":
                    metrics[
                        "completed"
                    ],

                "truncated":
                    metrics[
                        "truncated"
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

                "rolling_48_completion_rate":
                    rolling_completion_rate,

                "rolling_48_collection_fraction":
                    rolling_mean_collection,

                "rolling_48_invalid_action_rate":
                    rolling_mean_invalid,

                "rolling_48_reward":
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
                f" | order "
                f"{order}"
                f" | collect "
                f"{metrics['items_collected']}/2"
                f" | complete "
                f"{int(metrics['completed'])}"
                f" | dist "
                f"{metrics['distance']:3.0f}"
                f" | invalid "
                f"{metrics['invalid_action_rate']:5.1f}%"
                f" | roll complete "
                f"{rolling_completion_rate:5.1f}%"
                f" | roll collect "
                f"{rolling_mean_collection:.3f}"
                f" | eps "
                f"{metrics['epsilon']:.3f}"
                f" | loss "
                f"{loss_text}"
            )

        # ====================================================
        # DEVELOPMENT PROBE AFTER EACH COMPLETE CYCLE
        # ====================================================

        if (
            episode
            % evaluate_every
            == 0
        ):

            (
                probe_summary,
                probe_raw,
            ) = evaluate_probe_set(
                env,
                encoder,
                agent,
                probe_orders,
                probe_optima,
                all_pick_locations,
                episode=episode,
                cycle=cycle,
            )

            probe_summaries.append(
                probe_summary
            )

            probe_raw_results.extend(
                probe_raw
            )

            gap_value = (
                probe_summary[
                    "mean_optimality_gap"
                ]
            )

            if np.isfinite(
                gap_value
            ):

                gap_text = (
                    f"{gap_value:.2f}%"
                )

            else:

                gap_text = "N/A"

            print(
                "\n"
                "------------------------------------------------"
            )

            print(
                f"TWO-PICK DEVELOPMENT EVALUATION "
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
                f"{probe_summary['completion_rate']:.1f}%"
            )

            print(
                f"Completed probes:         "
                f"{probe_summary['completed_orders']}"
                f"/{probe_summary['total_orders']}"
            )

            print(
                f"Any-pick rate:            "
                f"{probe_summary['any_pick_rate']:.1f}%"
            )

            print(
                f"Mean items collected:     "
                f"{probe_summary['mean_items_collected']:.3f}"
                f"/2"
            )

            print(
                f"Mean collection fraction: "
                f"{probe_summary['mean_collection_fraction']:.3f}"
            )

            print(
                f"Mean invalid-action rate: "
                f"{probe_summary['mean_invalid_action_rate']:.1f}%"
            )

            print(
                f"Mean completed distance:  "
                f"{probe_summary['mean_completed_distance']}"
            )

            print(
                f"Mean optimality gap:      "
                f"{gap_text}"
            )

            # ------------------------------------------------
            # Best development checkpoint
            # ------------------------------------------------

            if probe_score_is_better(
                probe_summary,
                best_probe_summary,
            ):

                best_probe_summary = (
                    probe_summary.copy()
                )

                agent.save(
                    best_model_path
                )

                print(
                    "New best two-pick probe model saved."
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
                probe_summaries,
                probe_summary_path,
            )

            save_csv(
                probe_raw_results,
                probe_raw_path,
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
    # FINAL CSV SAVING
    # ========================================================

    save_csv(
        training_history,
        training_history_path,
    )

    save_csv(
        probe_summaries,
        probe_summary_path,
    )

    save_csv(
        probe_raw_results,
        probe_raw_path,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        "================================================\n"
        "BALANCED TWO-PICK TRAINING COMPLETE\n"
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
        f"Development combinations excluded "
        f"from training: {PROBE_ORDERS}"
    )

    print(
        f"Coordinate state dimension: "
        f"{encoder.output_dim}"
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
    # BEST DEVELOPMENT CHECKPOINT
    # ========================================================

    if best_probe_summary is not None:

        print(
            "\nBest two-pick development checkpoint:"
        )

        print(
            f"  Episode: "
            f"{best_probe_summary['episode']}"
        )

        print(
            f"  Cycle: "
            f"{best_probe_summary['cycle']}"
        )

        print(
            f"  Completion rate: "
            f"{best_probe_summary['completion_rate']:.1f}%"
        )

        print(
            f"  Any-pick rate: "
            f"{best_probe_summary['any_pick_rate']:.1f}%"
        )

        print(
            f"  Mean items collected: "
            f"{best_probe_summary['mean_items_collected']:.3f}"
            f"/2"
        )

        print(
            f"  Mean collection fraction: "
            f"{best_probe_summary['mean_collection_fraction']:.3f}"
        )

        print(
            f"  Mean invalid-action rate: "
            f"{best_probe_summary['mean_invalid_action_rate']:.1f}%"
        )

        best_gap = (
            best_probe_summary[
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

    # ========================================================
    # SAVED OUTPUTS
    # ========================================================

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Experiment config:\n"
        f"    {config_path}"
    )

    print(
        f"  Training schedule:\n"
        f"    {schedule_path}"
    )

    print(
        f"  Probe definitions:\n"
        f"    {probe_definition_path}"
    )

    print(
        f"  Training history:\n"
        f"    {training_history_path}"
    )

    print(
        f"  Probe summary:\n"
        f"    {probe_summary_path}"
    )

    print(
        f"  Probe raw results:\n"
        f"    {probe_raw_path}"
    )

    print(
        f"  Best probe model:\n"
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
