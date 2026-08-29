"""
Balanced one-pick DQN training experiment.

Purpose
-------
Train the movement-level DQN across all 96 warehouse pick
locations in a controlled and balanced way.

Each of the 96 pick locations is presented exactly once per
training cycle.

Training:
    96 locations
    x 30 cycles
    = 2880 training episodes

Within each cycle:
    - all 96 locations appear exactly once
    - their order is deterministically shuffled

This removes uneven target exposure as a possible explanation
for poor DQN learning.

The experiment uses:
    - potential-based reward shaping
    - corrected termination/truncation handling in DQNAgent
    - fixed one-pick probe set
    - exact optimal distances
    - greedy probe evaluation
    - best-checkpoint selection

Important
---------
The fixed probe locations also occur during training.

Therefore the probe set is NOT a held-out generalisation test.
It is a fixed development/probe set for monitoring learning.

Final experimental test orders must remain separate.
"""

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
# EXPERIMENT SETTINGS
# ============================================================

RUN_NAME = "balanced_one_pick_30_cycles"

TRAIN_SEED = 42

TRAIN_CYCLES = 30

ORDER_SIZE = 1

MAX_STEPS = 150


# ============================================================
# PROBE EVALUATION
# ============================================================

PROBE_ORDERS = 20

PROBE_SEED_START = 50_000

# One evaluation after every complete 96-location cycle.
EVALUATE_EVERY_CYCLES = 1


# ============================================================
# CONSOLE / ROLLING METRICS
# ============================================================

# Once warehouse size is known this will effectively be
# one full cycle = 96 episodes.
ROLLING_WINDOW = 96

# Print training status twice during each 96-location cycle.
PRINT_EVERY = 48


# ============================================================
# DQN HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 1e-4

GAMMA = 0.99

EPSILON_START = 1.0

EPSILON_MIN = 0.05

# Slower than the fixed-target diagnostic because the agent
# must learn 96 different target locations.
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

    Total fixed pick locations:
        8 aisles x 12 rows = 96
    """

    height = 14
    width = 17

    # 1 = obstacle
    # 0 = walkable
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

    depot = (0, 0)

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
    Create the warehouse routing environment.
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
# DQN
# ============================================================

def create_agent(env):
    """
    Create the DQN agent.
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
# SHORTEST-PATH DISTANCES
# ============================================================

def bfs_distance_map(
    grid,
    start,
):
    """
    Calculate shortest-path distances from one walkable cell
    to every reachable walkable warehouse cell.
    """

    rows, cols = grid.shape

    if grid[start] != 0:

        raise ValueError(
            f"BFS start position is not walkable: {start}"
        )

    distances = {
        start: 0
    }

    queue = deque(
        [start]
    )

    directions = [
        (-1, 0),   # up
        (1, 0),    # down
        (0, -1),   # left
        (0, 1),    # right
    ]

    while queue:

        current = queue.popleft()

        row, col = current

        current_distance = distances[
            current
        ]

        for dr, dc in directions:

            new_row = row + dr
            new_col = col + dc

            if not (
                0 <= new_row < rows
                and
                0 <= new_col < cols
            ):
                continue

            if grid[
                new_row,
                new_col
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
    Pre-compute shortest-path maps from the depot and all
    96 possible pick locations.
    """

    relevant_points = [
        depot,
        *all_pick_locations,
    ]

    distance_lookup = {}

    for point in relevant_points:

        distance_lookup[
            point
        ] = bfs_distance_map(
            grid,
            point,
        )

    return distance_lookup


# ============================================================
# BALANCED TRAINING SCHEDULE
# ============================================================

def build_balanced_training_schedule(
    all_pick_locations,
):
    """
    Build the complete balanced one-pick training schedule.

    Every cycle contains every pick location exactly once.

    Each cycle uses its own deterministic shuffle.

    Returns
    -------
    list
        A list of dictionaries containing:

            episode
            cycle
            cycle_position
            target
            order
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

        # Use a local RNG so building the schedule does not
        # alter the global DQN exploration random stream.
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
    Verify that every one of the 96 locations appears exactly
    TRAIN_CYCLES times in the complete training schedule.
    """

    expected_episodes = (
        len(all_pick_locations)
        * TRAIN_CYCLES
    )

    if len(schedule) != expected_episodes:

        raise AssertionError(
            "Balanced schedule has incorrect length: "
            f"{len(schedule)} != {expected_episodes}"
        )

    target_counts = Counter(
        row["target"]
        for row in schedule
    )

    if len(target_counts) != len(
        all_pick_locations
    ):

        raise AssertionError(
            "Balanced schedule does not contain all "
            "pick locations."
        )

    for location in all_pick_locations:

        count = target_counts[
            location
        ]

        if count != TRAIN_CYCLES:

            raise AssertionError(
                f"Location {location} appears "
                f"{count} times instead of "
                f"{TRAIN_CYCLES}."
            )

    # Also validate each individual cycle.
    for cycle_number in range(
        1,
        TRAIN_CYCLES + 1,
    ):

        cycle_locations = [
            row["target"]
            for row in schedule
            if row["cycle"]
            == cycle_number
        ]

        if len(
            cycle_locations
        ) != len(
            all_pick_locations
        ):

            raise AssertionError(
                f"Cycle {cycle_number} does not contain "
                "exactly 96 episodes."
            )

        if set(
            cycle_locations
        ) != set(
            all_pick_locations
        ):

            raise AssertionError(
                f"Cycle {cycle_number} does not contain "
                "every pick location exactly once."
            )


# ============================================================
# PROBE ORDERS
# ============================================================

def order_key(order):
    """
    Convert order into a stable hashable representation.
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
    Create the same deterministic 20 one-pick probe orders
    used to monitor learning.

    These are development probes, not held-out test orders.
    """

    orders = []

    seen = set()

    seed = PROBE_SEED_START

    while len(orders) < PROBE_ORDERS:

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
# STATE DECODING
# ============================================================

def decode_agent_position(
    state,
    grid,
):
    """
    Recover the picker coordinate from the first two
    normalised state values.
    """

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    rows, cols = grid.shape

    row = int(
        round(
            float(state[0])
            * (rows - 1)
        )
    )

    col = int(
        round(
            float(state[1])
            * (cols - 1)
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
    state,
    all_pick_locations,
):
    """
    Recover remaining required pick locations from the
    binary part of the DQN state.
    """

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    remaining_bits = state[2:]

    if len(
        remaining_bits
    ) != len(
        all_pick_locations
    ):

        raise ValueError(
            "State pick-vector length does not match "
            "all_pick_locations."
        )

    remaining_locations = []

    for (
        index,
        bit,
    ) in enumerate(
        remaining_bits
    ):

        if bit > 0.5:

            remaining_locations.append(
                all_pick_locations[index]
            )

    return remaining_locations


# ============================================================
# COLLECTION METRICS
# ============================================================

def get_collection_metrics(
    state,
    order_size,
):
    """
    Calculate collection progress from the DQN state.
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
    Calculate valid/invalid actions.

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
        int(steps)
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
    state,
    grid,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Calculate one-pick routing potential.

    Before collection:

        Phi(s) =
        -(distance picker -> target
          + distance target -> depot)

    After collection:

        Phi(s) =
        -distance picker -> depot
    """

    agent_position = (
        decode_agent_position(
            state,
            grid,
        )
    )

    remaining_picks = (
        get_remaining_pick_locations(
            state,
            all_pick_locations,
        )
    )

    # --------------------------------------------------------
    # Pick has already been collected
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
    # One-pick experiment
    # --------------------------------------------------------

    if len(
        remaining_picks
    ) != 1:

        raise ValueError(
            "Balanced one-pick experiment expected zero "
            "or one remaining pick."
        )

    target = remaining_picks[0]

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
    state,
    next_state,
    grid,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Potential-based reward shaping:

        F(s,s') =
        scale *
        [gamma * Phi(s') - Phi(s)]
    """

    current_potential = (
        calculate_potential(
            state,
            grid,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    next_potential = (
        calculate_potential(
            next_state,
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
    agent,
    order,
    grid,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Train the DQN for one complete one-pick episode.
    """

    state = env.reset(
        order
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
            state
        )

        # ----------------------------------------------------
        # Environment transition
        # ----------------------------------------------------

        (
            next_state,
            base_reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        # ----------------------------------------------------
        # Potential-based shaping
        # ----------------------------------------------------

        shaping_reward = (
            calculate_shaping_reward(
                state,
                next_state,
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
        # Replay
        #
        # DQNAgent now handles terminated/truncated
        # separately.
        # ----------------------------------------------------

        agent.remember(
            state,
            action,
            training_reward,
            next_state,
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

        state = next_state

    # --------------------------------------------------------
    # Episode diagnostics
    # --------------------------------------------------------

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        state,
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
    agent,
    order,
    optimal_distance,
):
    """
    Evaluate one order with a purely greedy DQN policy.

    No replay or gradient updates occur.
    """

    state = env.reset(
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
            state,
            eval_mode=True,
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

    runtime = (
        time.perf_counter()
        - start_time
    )

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        state,
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
            float(
                optimality_gap
            )
            if np.isfinite(
                optimality_gap
            )
            else np.nan,

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
# PROBE OPTIMAL DISTANCES
# ============================================================

def calculate_probe_optima(
    grid,
    depot,
    probe_orders,
):
    """
    Calculate exact optimal distances for all fixed probes.
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
# PROBE SET EVALUATION
# ============================================================

def evaluate_probe_set(
    env,
    agent,
    probe_orders,
    probe_optima,
    episode,
    cycle,
):
    """
    Evaluate the greedy DQN on all fixed one-pick probes.
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

    # --------------------------------------------------------
    # Invalid actions across all probes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Completed-route metrics
    # --------------------------------------------------------

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
# BEST CHECKPOINT SELECTION
# ============================================================

def probe_score_is_better(
    new_summary,
    best_summary,
):
    """
    Select the best development checkpoint.

    Priority:
        1. higher completion rate
        2. higher collection fraction
        3. lower completed-route optimality gap
        4. lower invalid-action rate
    """

    if best_summary is None:
        return True

    # --------------------------------------------------------
    # 1. Completion
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
    # 2. Collection
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
    # 3. Optimality gap
    # --------------------------------------------------------

    new_gap = new_summary[
        "mean_optimality_gap"
    ]

    best_gap = best_summary[
        "mean_optimality_gap"
    ]

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
    Save a list of dictionary records to CSV.
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
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "================================================\n"
        "DQN BALANCED ONE-PICK TRAINING\n"
        "================================================"
    )

    # ========================================================
    # SEEDS
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

    print(
        f"Run name:              {RUN_NAME}\n"
        f"Training cycles:       {TRAIN_CYCLES}\n"
        f"Locations per cycle:   {location_count}\n"
        f"Training episodes:     {total_training_episodes}\n"
        f"Order size:            {ORDER_SIZE}\n"
        f"Probe orders:          {PROBE_ORDERS}\n"
        f"Evaluate every:        {evaluate_every} episodes\n"
        f"Max episode steps:     {MAX_STEPS}\n"
        f"Training seed:         {TRAIN_SEED}\n"
        f"Epsilon decay:         {EPSILON_DECAY}\n"
        f"Shaping scale:         {SHAPING_SCALE}\n"
        f"State dimension:       {2 + location_count}\n"
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
    # DISTANCE LOOKUP
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
    # PROBE SET
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
    # SAVE EXACT TRAINING SCHEDULE
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
    # HISTORY
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
    # INITIAL GREEDY PROBE
    # ========================================================

    print(
        "\nEvaluating untrained policy..."
    )

    (
        initial_summary,
        initial_raw,
    ) = evaluate_probe_set(
        env,
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

        episode = schedule_row[
            "episode"
        ]

        cycle = schedule_row[
            "cycle"
        ]

        cycle_position = schedule_row[
            "cycle_position"
        ]

        target = schedule_row[
            "target"
        ]

        order = schedule_row[
            "order"
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
        # PROBE EVALUATION AFTER EACH COMPLETE CYCLE
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

            gap_value = probe_summary[
                "mean_optimality_gap"
            ]

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
                f"PROBE EVALUATION AFTER CYCLE "
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
                f"Mean invalid-action rate:"
                f" "
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
            # BEST MODEL
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
                    "New best probe model saved."
                )

            print()

            # ------------------------------------------------
            # SAVE INTERMEDIATE OUTPUTS
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

    total_training_time = (
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
    # SAVE FINAL CSV DATA
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
        "BALANCED ONE-PICK TRAINING COMPLETE\n"
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
    # BEST PROBE CHECKPOINT
    # ========================================================

    if best_probe_summary is not None:

        print(
            "\nBest probe checkpoint:"
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

        best_gap = best_probe_summary[
            "mean_optimality_gap"
        ]

        if np.isfinite(
            best_gap
        ):

            print(
                f"  Mean optimality gap: "
                f"{best_gap:.2f}%"
            )

    # ========================================================
    # FILES
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
