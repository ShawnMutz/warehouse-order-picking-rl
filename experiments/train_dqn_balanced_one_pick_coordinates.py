"""
Balanced one-pick DQN experiment using coordinate-based state encoding.

Purpose
-------
Repeat the balanced one-pick training experiment while replacing
the original 98-dimensional binary-location DQN input with the new
62-dimensional coordinate-based representation.

The underlying WarehouseEnv is NOT changed.

WarehouseEnv still produces:

    2 picker coordinates
    +
    96 binary remaining-pick indicators

The CoordinateStateEncoder converts this into:

    2 picker coordinates
    +
    20 pick slots x 3 values

Each pick slot contains:

    normalised pick row
    normalised pick column
    active flag

Total DQN state dimension:

    2 + (20 * 3) = 62

Training schedule
-----------------
96 possible one-pick targets
x
30 balanced cycles
=
2880 training episodes

Every target appears exactly once during each cycle.

Comparison
----------
This experiment is intended to be compared directly against:

    balanced_one_pick_30_cycles

The principal experimental change is the DQN state representation.
"""

from utils.state_encoding import CoordinateStateEncoder
from utils.order_generation import generate_order
from exact.ratliff_rosenthal import exact_optimal_distance
from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent
import csv
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

RUN_NAME = "balanced_one_pick_coordinates_30_cycles"

TRAIN_SEED = 42

TRAIN_CYCLES = 30

ORDER_SIZE = 1

MAX_ORDER_SIZE = 20

MAX_STEPS = 150


# ============================================================
# PROBE EVALUATION
# ============================================================

PROBE_ORDERS = 20

# Keep identical to the previous balanced binary-state
# experiment so the probe targets are directly comparable.
PROBE_SEED_START = 50_000

EVALUATE_EVERY_CYCLES = 1


# ============================================================
# CONSOLE / ROLLING METRICS
# ============================================================

ROLLING_WINDOW = 96

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

    Total locations:
        8 x 12 = 96
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

def set_random_seeds(seed):
    """
    Set deterministic random seeds.
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
# STATE ENCODER
# ============================================================

def create_state_encoder(
    grid,
    all_pick_locations,
):
    """
    Create the coordinate-based state encoder.

    Output dimension:

        2 + (20 * 3) = 62
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
    Create DQN using the coordinate-state input dimension.
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
# SHORTEST PATHS
# ============================================================

def bfs_distance_map(
    grid,
    start,
):
    """
    Compute shortest path distance from start to every
    reachable walkable warehouse coordinate.
    """

    rows, cols = grid.shape

    if grid[start] != 0:
        raise ValueError(
            f"BFS start is not walkable: {start}"
        )

    distances = {
        start: 0
    }

    queue = deque(
        [start]
    )

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    while queue:

        current = queue.popleft()

        row, col = current

        current_distance = distances[
            current
        ]

        for dr, dc in directions:

            new_row = (
                row
                + dr
            )

            new_col = (
                col
                + dc
            )

            if not (
                0 <= new_row < rows
                and
                0 <= new_col < cols
            ):
                continue

            if grid[
                new_row,
                new_col,
            ] != 0:
                continue

            new_position = (
                new_row,
                new_col,
            )

            if new_position in distances:
                continue

            distances[
                new_position
            ] = (
                current_distance
                + 1
            )

            queue.append(
                new_position
            )

    return distances


def build_distance_lookup(
    grid,
    depot,
    all_pick_locations,
):
    """
    Pre-compute distances required by potential shaping.
    """

    relevant_points = [
        depot,
        *all_pick_locations,
    ]

    lookup = {}

    for point in relevant_points:

        lookup[
            point
        ] = bfs_distance_map(
            grid,
            point,
        )

    return lookup


# ============================================================
# BALANCED TRAINING SCHEDULE
# ============================================================

def build_balanced_training_schedule(
    all_pick_locations,
):
    """
    Build 30 balanced training cycles.

    Every target appears exactly once per cycle.
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

        locations = list(
            all_pick_locations
        )

        # Local RNG prevents schedule generation from
        # consuming the global exploration random stream.
        cycle_rng = random.Random(
            TRAIN_SEED
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

                    "target":
                        target,

                    "order":
                        [target],
                }
            )

    return schedule


def validate_balanced_training_schedule(
    schedule,
    all_pick_locations,
):
    """
    Confirm every location receives exactly 30 exposures.
    """

    expected_episodes = (
        len(
            all_pick_locations
        )
        * TRAIN_CYCLES
    )

    if len(
        schedule
    ) != expected_episodes:

        raise AssertionError(
            "Incorrect schedule length."
        )

    target_counts = Counter(
        row[
            "target"
        ]
        for row in schedule
    )

    if len(
        target_counts
    ) != len(
        all_pick_locations
    ):

        raise AssertionError(
            "Schedule does not include all target locations."
        )

    for location in all_pick_locations:

        if target_counts[
            location
        ] != TRAIN_CYCLES:

            raise AssertionError(
                f"{location} appears "
                f"{target_counts[location]} times instead "
                f"of {TRAIN_CYCLES}."
            )

    # Validate each individual cycle.
    for cycle_number in range(
        1,
        TRAIN_CYCLES + 1,
    ):

        cycle_locations = [
            row[
                "target"
            ]
            for row in schedule
            if row[
                "cycle"
            ] == cycle_number
        ]

        if len(
            cycle_locations
        ) != len(
            all_pick_locations
        ):

            raise AssertionError(
                f"Cycle {cycle_number} has incorrect length."
            )

        if set(
            cycle_locations
        ) != set(
            all_pick_locations
        ):

            raise AssertionError(
                f"Cycle {cycle_number} does not contain all "
                "locations exactly once."
            )


# ============================================================
# PROBE ORDERS
# ============================================================

def order_key(
    order,
):
    """
    Stable hashable representation of an order.
    """

    return tuple(
        sorted(
            order
        )
    )


def create_probe_orders(
    all_pick_locations,
    aisle_columns,
):
    """
    Generate the same deterministic 20 one-pick probe orders
    as the previous balanced experiment.

    These are development probes, NOT held-out test orders.
    """

    orders = []

    seen = set()

    seed = PROBE_SEED_START

    while len(
        orders
    ) < PROBE_ORDERS:

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=1,
            distribution="uniform",
            seed=seed,
            aisle_columns=aisle_columns,
        )

        key = order_key(
            order
        )

        if key not in seen:

            orders.append(
                order
            )

            seen.add(
                key
            )

        seed += 1

    return orders


# ============================================================
# RAW ENVIRONMENT STATE DECODING
# ============================================================

def decode_agent_position(
    environment_state,
    grid,
):
    """
    Recover the discrete picker coordinate from the RAW
    WarehouseEnv state.

    Important:
        This function uses the 98-dimensional environment
        state, NOT the 62-dimensional encoded DQN state.
    """

    environment_state = np.asarray(
        environment_state,
        dtype=np.float32,
    )

    rows, cols = grid.shape

    row = int(
        round(
            float(
                environment_state[0]
            )
            * (
                rows - 1
            )
        )
    )

    col = int(
        round(
            float(
                environment_state[1]
            )
            * (
                cols - 1
            )
        )
    )

    row = max(
        0,
        min(
            rows - 1,
            row,
        ),
    )

    col = max(
        0,
        min(
            cols - 1,
            col,
        ),
    )

    return (
        row,
        col,
    )


def get_remaining_pick_locations(
    environment_state,
    all_pick_locations,
):
    """
    Decode the remaining picks from the RAW WarehouseEnv
    98-dimensional state.

    WarehouseEnv stores the pick indicators in sorted
    coordinate order.
    """

    environment_state = np.asarray(
        environment_state,
        dtype=np.float32,
    )

    remaining_bits = (
        environment_state[
            2:
        ]
    )

    state_locations = sorted(
        tuple(
            location
        )
        for location in all_pick_locations
    )

    if len(
        remaining_bits
    ) != len(
        state_locations
    ):

        raise ValueError(
            "Raw environment state does not contain the "
            "expected number of pick indicators."
        )

    remaining_locations = []

    for (
        location,
        bit,
    ) in zip(
        state_locations,
        remaining_bits,
    ):

        if bit > 0.5:

            remaining_locations.append(
                location
            )

    return remaining_locations


# ============================================================
# COLLECTION METRICS
# ============================================================

def get_collection_metrics(
    environment_state,
    order_size,
):
    """
    Calculate collection progress from RAW environment state.
    """

    environment_state = np.asarray(
        environment_state,
        dtype=np.float32,
    )

    remaining_bits = (
        environment_state[
            2:
        ]
    )

    items_remaining = int(
        np.count_nonzero(
            remaining_bits > 0.5
        )
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
    Every valid movement increases travel distance by 1.

    Therefore:

        invalid actions =
            steps - travel distance
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
# POTENTIAL FUNCTION
# ============================================================

def calculate_potential(
    environment_state,
    grid,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Calculate one-pick routing potential from the RAW
    environment state.

    Before collection:

        Phi(s) =
        -[
            distance(current, target)
            +
            distance(target, depot)
         ]

    After collection:

        Phi(s) =
        -distance(current, depot)
    """

    agent_position = decode_agent_position(
        environment_state,
        grid,
    )

    remaining_picks = (
        get_remaining_pick_locations(
            environment_state,
            all_pick_locations,
        )
    )

    # --------------------------------------------------------
    # Pick already collected: objective is return to depot.
    # --------------------------------------------------------

    if len(
        remaining_picks
    ) == 0:

        distance_to_depot = (
            distance_lookup[
                depot
            ][
                agent_position
            ]
        )

        return -float(
            distance_to_depot
        )

    # --------------------------------------------------------
    # This experiment should never have >1 remaining pick.
    # --------------------------------------------------------

    if len(
        remaining_picks
    ) != 1:

        raise ValueError(
            "Balanced coordinate one-pick experiment expected "
            "zero or one remaining pick."
        )

    target = (
        remaining_picks[0]
    )

    distance_to_target = (
        distance_lookup[
            target
        ][
            agent_position
        ]
    )

    target_to_depot = (
        distance_lookup[
            target
        ][
            depot
        ]
    )

    estimated_remaining_route = (
        distance_to_target
        + target_to_depot
    )

    return -float(
        estimated_remaining_route
    )


# ============================================================
# POTENTIAL-BASED SHAPING
# ============================================================

def calculate_shaping_reward(
    environment_state,
    next_environment_state,
    grid,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Potential-based shaping:

        F(s, s') =
            scale *
            [gamma * Phi(s') - Phi(s)]

    Potentials are calculated using RAW WarehouseEnv states.
    """

    current_potential = (
        calculate_potential(
            environment_state,
            grid,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    next_potential = (
        calculate_potential(
            next_environment_state,
            grid,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    shaping_reward = (
        SHAPING_SCALE
        * (
            GAMMA
            * next_potential
            - current_potential
        )
    )

    return float(
        shaping_reward
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
    Train for one coordinate-state one-pick episode.

    Important separation:

        environment_state
            = original 98-dimensional WarehouseEnv state

        dqn_state
            = encoded 62-dimensional coordinate state

    Reward shaping and collection metrics use the raw state.

    DQN action selection and replay use the encoded state.
    """

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    environment_state = (
        env.reset(
            order
        )
    )

    # --------------------------------------------------------
    # Configure fixed coordinate slots for this order
    # --------------------------------------------------------

    encoder.reset(
        order
    )

    # --------------------------------------------------------
    # Encode state for DQN
    # --------------------------------------------------------

    dqn_state = encoder.encode(
        environment_state
    )

    if dqn_state.shape != (
        encoder.output_dim,
    ):

        raise RuntimeError(
            "Unexpected encoded state dimension."
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
        # Select action from 62-dimensional state
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
        # Reward shaping from raw state
        # ----------------------------------------------------

        shaping_reward = (
            calculate_shaping_reward(
                environment_state,
                next_environment_state,
                grid,
                depot,
                all_pick_locations,
                distance_lookup,
            )
        )

        training_reward = (
            float(
                base_reward
            )
            + shaping_reward
        )

        # ----------------------------------------------------
        # Encode next state for DQN
        # ----------------------------------------------------

        next_dqn_state = (
            encoder.encode(
                next_environment_state
            )
        )

        # ----------------------------------------------------
        # Replay uses encoded states
        # ----------------------------------------------------

        agent.remember(
            dqn_state,
            action,
            training_reward,
            next_dqn_state,
            terminated,
            truncated,
        )

        loss = agent.train_step()

        if loss is not None:

            if not np.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite DQN loss: {loss}"
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
    # Episode metrics from raw final state
    # --------------------------------------------------------

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        environment_state,
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
# GREEDY EVALUATION
# ============================================================

def evaluate_dqn_order(
    env,
    encoder,
    agent,
    order,
    optimal_distance,
):
    """
    Evaluate one order using the greedy coordinate-state DQN.
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

        next_dqn_state = (
            encoder.encode(
                next_environment_state
            )
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
# PROBE OPTIMA
# ============================================================

def calculate_probe_optima(
    grid,
    depot,
    probe_orders,
):
    """
    Calculate exact optimal distances for all probe orders.
    """

    optima = []

    print(
        "\nCalculating probe-set optimal distances..."
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
# PROBE EVALUATION
# ============================================================

def evaluate_probe_set(
    env,
    encoder,
    agent,
    probe_orders,
    probe_optima,
    episode,
    cycle,
):
    """
    Evaluate greedy policy on all fixed development probes.
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
                        sorted(
                            order
                        )
                    ),

                **evaluation,
            }
        )

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
                for row
                in raw_results
            ]
        )
    )

    mean_items_collected = float(
        np.mean(
            [
                row[
                    "items_collected"
                ]
                for row
                in raw_results
            ]
        )
    )

    mean_invalid_action_rate = float(
        np.mean(
            [
                row[
                    "invalid_action_rate"
                ]
                for row
                in raw_results
            ]
        )
    )

    if completed_results:

        mean_completed_distance = float(
            np.mean(
                [
                    row[
                        "distance"
                    ]
                    for row
                    in completed_results
                ]
            )
        )

        mean_optimality_gap = float(
            np.mean(
                [
                    row[
                        "optimality_gap"
                    ]
                    for row
                    in completed_results
                ]
            )
        )

        mean_completed_steps = float(
            np.mean(
                [
                    row[
                        "steps"
                    ]
                    for row
                    in completed_results
                ]
            )
        )

    else:

        mean_completed_distance = np.nan

        mean_optimality_gap = np.nan

        mean_completed_steps = np.nan

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

        "mean_optimality_gap":
            mean_optimality_gap,

        "mean_completed_steps":
            mean_completed_steps,
    }

    return (
        summary,
        raw_results,
    )


# ============================================================
# BEST CHECKPOINT
# ============================================================

def probe_score_is_better(
    new_summary,
    best_summary,
):
    """
    Development checkpoint priority:

        1. higher completion rate
        2. higher collection fraction
        3. lower optimality gap
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
    # Optimality
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
    Save list of dictionaries to CSV.
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
        "DQN BALANCED ONE-PICK COORDINATE TRAINING\n"
        "================================================"
    )

    # ========================================================
    # SEED
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

    total_training_episodes = (
        location_count
        * TRAIN_CYCLES
    )

    evaluate_every = (
        location_count
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

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if encoder.output_dim != 62:

        raise RuntimeError(
            f"Expected coordinate state dimension 62, "
            f"received {encoder.output_dim}."
        )

    if agent.input_dim != 62:

        raise RuntimeError(
            f"Expected DQN input dimension 62, "
            f"received {agent.input_dim}."
        )

    print(
        f"Run name:              {RUN_NAME}\n"
        f"Training cycles:       {TRAIN_CYCLES}\n"
        f"Locations per cycle:   {location_count}\n"
        f"Training episodes:     {total_training_episodes}\n"
        f"Order size:            {ORDER_SIZE}\n"
        f"Maximum encoded picks: {MAX_ORDER_SIZE}\n"
        f"Probe orders:          {PROBE_ORDERS}\n"
        f"Evaluate every:        {evaluate_every} episodes\n"
        f"Max episode steps:     {MAX_STEPS}\n"
        f"Training seed:         {TRAIN_SEED}\n"
        f"Epsilon decay:         {EPSILON_DECAY}\n"
        f"Shaping scale:         {SHAPING_SCALE}\n"
        f"Raw env state dim:     {2 + location_count}\n"
        f"Encoded DQN state dim: {encoder.output_dim}\n"
    )

    # ========================================================
    # BALANCED SCHEDULE
    # ========================================================

    print(
        "Building balanced training schedule..."
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
        f"  {location_count} locations"
    )

    print(
        f"  {TRAIN_CYCLES} exposures per location"
    )

    print(
        f"  {len(training_schedule)} total episodes"
    )

    # ========================================================
    # DISTANCES
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
    # PROBES
    # ========================================================

    probe_orders = (
        create_probe_orders(
            all_pick_locations,
            aisle_columns,
        )
    )

    probe_optima = (
        calculate_probe_optima(
            grid,
            depot,
            probe_orders,
        )
    )

    # ========================================================
    # OUTPUT DIRECTORIES
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
    # SAVE SCHEDULE
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

            "target":
                str(
                    row[
                        "target"
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
                    sorted(
                        order
                    )
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
    # HISTORIES
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
    # UNTRAINED PROBE
    # ========================================================

    print(
        "\nEvaluating untrained coordinate policy..."
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

        target = (
            schedule_row[
                "target"
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
        # Rolling statistics
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

        rolling_mean_invalid_rate = float(
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
        # Training record
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

                "rolling_96_completion_rate":
                    rolling_completion_rate,

                "rolling_96_collection_fraction":
                    rolling_mean_collection,

                "rolling_96_invalid_action_rate":
                    rolling_mean_invalid_rate,

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
                f"{cycle_position:02d}/{location_count}"
                f" | target "
                f"{target}"
                f" | collect "
                f"{metrics['items_collected']}/1"
                f" | complete "
                f"{int(metrics['completed'])}"
                f" | distance "
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
        # PROBE AFTER EACH COMPLETE CYCLE
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
                f"COORDINATE PROBE EVALUATION AFTER CYCLE "
                f"{cycle}/{TRAIN_CYCLES}"
            )

            print(
                "------------------------------------------------"
            )

            print(
                f"Episode:                 "
                f"{episode}"
            )

            print(
                f"Completion rate:         "
                f"{probe_summary['completion_rate']:.1f}%"
            )

            print(
                f"Completed probes:        "
                f"{probe_summary['completed_orders']}"
                f"/{probe_summary['total_orders']}"
            )

            print(
                f"Any-pick rate:           "
                f"{probe_summary['any_pick_rate']:.1f}%"
            )

            print(
                f"Mean collection frac:    "
                f"{probe_summary['mean_collection_fraction']:.3f}"
            )

            print(
                f"Mean invalid-action rate: "
                f"{probe_summary['mean_invalid_action_rate']:.1f}%"
            )

            print(
                f"Mean completed distance: "
                f"{probe_summary['mean_completed_distance']}"
            )

            print(
                f"Mean optimality gap:     "
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
                    "New best coordinate probe model saved."
                )

            print()

            # ------------------------------------------------
            # Save intermediate results
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
    # COMPLETE
    # ========================================================

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    agent.save(
        final_model_path
    )

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

    print(
        "\n"
        "================================================\n"
        "BALANCED COORDINATE ONE-PICK TRAINING COMPLETE\n"
        "================================================"
    )

    print(
        f"Training time: "
        f"{total_training_time:.2f} seconds"
    )

    print(
        f"Training episodes: "
        f"{total_training_episodes}"
    )

    print(
        f"Exposure per target: "
        f"{TRAIN_CYCLES}"
    )

    print(
        f"Coordinate DQN state dimension: "
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
    # BEST CHECKPOINT
    # ========================================================

    if best_probe_summary is not None:

        print(
            "\nBest coordinate probe checkpoint:"
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
    # OUTPUTS
    # ========================================================

    print(
        "\nSaved outputs:"
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
