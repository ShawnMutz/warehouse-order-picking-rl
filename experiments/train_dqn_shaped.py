"""
DQN one-pick potential-based reward-shaping experiment.

Purpose
-------
Test whether potential-based distance reward shaping improves
the ability of a movement-level DQN to learn warehouse routing.

This experiment follows the earlier one-pick learnability run.

Training:
    - 1-pick orders
    - uniform distribution
    - all 96 pick locations may appear during training
    - original environment reward retained
    - potential-based shaping added to training reward

Development evaluation:
    - fixed 1-pick orders
    - greedy DQN policy
    - no replay-buffer updates
    - no gradient updates
    - evaluated using actual travel distance
    - compared with exact optimal distance

Potential:
    Before collection:
        -(distance picker -> pick + distance pick -> depot)

    After collection:
        -distance picker -> depot

Shaping:
    F(s, s') = scale * (gamma * Phi(s') - Phi(s))

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


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RUN_NAME = "one_pick_potential_shaping"

TRAIN_SEED = 42

TRAIN_EPISODES = 500

ORDER_SIZE = 1

TRAIN_DISTRIBUTION = "uniform"

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

EPSILON_DECAY = 0.999995

BATCH_SIZE = 64

BUFFER_CAPACITY = 50_000

TARGET_UPDATE = 1_000

HIDDEN_DIM = 256


# ============================================================
# REWARD SHAPING SETTINGS
# ============================================================

SHAPING_SCALE = 1.0


# ============================================================
# WAREHOUSE CREATION
# ============================================================

def create_warehouse():
    """
    Create the fixed 14 x 17 warehouse.

    Layout
    ------
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

    # Vertical picking aisles
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
    Set random seeds for Python, NumPy and PyTorch.
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
    Create the warehouse environment.

    The original environment reward remains unchanged.
    Potential shaping is added only inside the DQN training
    loop.
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
# DQN AGENT
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
# SHORTEST PATH CALCULATION
# ============================================================

def bfs_distance_map(
    grid,
    start,
):
    """
    Calculate shortest-path distance from one walkable cell
    to every reachable walkable warehouse cell.

    Parameters
    ----------
    grid : np.ndarray
        Warehouse layout.

    start : tuple
        Starting (row, column).

    Returns
    -------
    dict
        position -> shortest-path distance
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
    Pre-compute shortest-path distance maps from the depot
    and every possible pick location.

    This prevents running BFS during every DQN transition.
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
    Create the fixed development evaluation set.

    For this one-pick learnability experiment, these locations
    are allowed to appear during training.

    The development evaluation itself never updates the DQN.
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


def generate_training_order(
    episode,
    all_pick_locations,
    aisle_columns,
):
    """
    Generate one deterministic training order.

    All 96 storage locations are eligible.
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
# STATE DECODING
# ============================================================

def decode_agent_position(
    state,
    grid,
):
    """
    Recover the discrete picker position from the first two
    normalised DQN state values.
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
    Decode required pick locations from the binary portion
    of the state vector.
    """

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    remaining_bits = state[2:]

    if len(remaining_bits) != len(
        all_pick_locations
    ):

        raise ValueError(
            "State pick-vector length does not match "
            "the fixed pick-location list."
        )

    remaining_locations = []

    for index, value in enumerate(
        remaining_bits
    ):

        if value > 0.5:

            remaining_locations.append(
                all_pick_locations[index]
            )

    return remaining_locations


# ============================================================
# COLLECTION DIAGNOSTICS
# ============================================================

def get_collection_metrics(
    state,
    order_size,
):
    """
    Calculate how much of the order has been collected using
    the binary remaining-pick vector.
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
        float(collection_fraction),
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
    Calculate the routing potential Phi(s).

    Before collection:
        Phi(s) =
            -(distance picker -> target
              + distance target -> depot)

    After collection:
        Phi(s) =
            -distance picker -> depot

    Higher values indicate progress toward completing the
    routing task.
    """

    agent_position = decode_agent_position(
        state,
        grid,
    )

    remaining_picks = (
        get_remaining_pick_locations(
            state,
            all_pick_locations,
        )
    )

    # --------------------------------------------------------
    # No remaining items:
    # objective is return to the depot.
    # --------------------------------------------------------

    if len(remaining_picks) == 0:

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
    # This experiment is specifically one-pick.
    # --------------------------------------------------------

    if len(remaining_picks) != 1:

        raise ValueError(
            "One-pick shaping experiment expected either "
            "zero or one remaining pick, but found "
            f"{len(remaining_picks)}."
        )

    target = remaining_picks[0]

    # Distance from current picker position to target.
    distance_to_target = (
        distance_lookup[
            target
        ][
            agent_position
        ]
    )

    # Distance from target back to depot.
    distance_target_to_depot = (
        distance_lookup[
            target
        ][
            depot
        ]
    )

    remaining_route_distance = (
        distance_to_target
        + distance_target_to_depot
    )

    return -float(
        remaining_route_distance
    )


# ============================================================
# POTENTIAL-BASED SHAPING REWARD
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
    Calculate potential-based shaping reward:

        F(s, s') =
            scale *
            (gamma * Phi(s') - Phi(s))
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
# GREEDY ACTION SELECTION
# ============================================================

def select_greedy_action(
    agent,
    state,
):
    """
    Select the highest-Q action directly from the policy
    network.

    This avoids changing epsilon or any exploration state
    during development evaluation.
    """

    device = next(
        agent.policy_net.parameters()
    ).device

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = agent.policy_net(
            state_tensor
        )

        action = int(
            torch.argmax(
                q_values,
                dim=1,
            ).item()
        )

    return action


# ============================================================
# TRAIN ONE EPISODE
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
    Run one complete shaped-reward DQN training episode.
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
        # Potential-based reward shaping
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
            float(base_reward)
            + shaping_reward
        )

        # ----------------------------------------------------
        # Replay buffer receives SHAPED reward
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
        # DQN gradient update
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
                float(loss)
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

    order_size = len(
        order
    )

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
            np.mean(
                losses
            )
        )
        if losses
        else np.nan
    )

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

        "completed":
            bool(
                terminated
            ),

        "truncated":
            bool(
                truncated
            ),

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
# GREEDY DQN EVALUATION
# ============================================================

def evaluate_dqn_order(
    env,
    agent,
    order,
):
    """
    Evaluate the current policy greedily.

    Important:
        - no shaped reward is required for evaluation
        - no replay memory
        - no gradient updates
        - actual travel distance is measured
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

        action = select_greedy_action(
            agent,
            state,
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

    order_size = len(
        order
    )

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

        "steps":
            int(
                env.steps
            ),

        "runtime":
            float(
                runtime
            ),

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
# DEVELOPMENT OPTIMAL DISTANCES
# ============================================================

def calculate_development_optima(
    grid,
    depot,
    development_orders,
):
    """
    Calculate exact optimal route distances for the fixed
    development set.
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
            float(
                optimum
            )
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
    Evaluate the greedy policy over the complete development
    set.
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

        if completed:

            optimality_gap = (
                (
                    evaluation[
                        "distance"
                    ]
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
                        sorted(
                            order
                        )
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
        if result[
            "completed"
        ]
    ]

    completion_rate = (
        100.0
        * len(
            completed_results
        )
        / len(
            results
        )
    )

    # --------------------------------------------------------
    # Pick collection metrics
    # --------------------------------------------------------

    orders_collecting_any_pick = sum(
        result[
            "items_collected"
        ] > 0
        for result in results
    )

    any_pick_rate = (
        100.0
        * orders_collecting_any_pick
        / len(
            results
        )
    )

    mean_items_collected = float(
        np.mean(
            [
                result[
                    "items_collected"
                ]
                for result
                in results
            ]
        )
    )

    mean_collection_fraction = float(
        np.mean(
            [
                result[
                    "collection_fraction"
                ]
                for result
                in results
            ]
        )
    )

    # --------------------------------------------------------
    # Successfully completed orders only
    # --------------------------------------------------------

    if completed_results:

        mean_distance = float(
            np.mean(
                [
                    result[
                        "distance"
                    ]
                    for result
                    in completed_results
                ]
            )
        )

        mean_gap = float(
            np.mean(
                [
                    result[
                        "optimality_gap"
                    ]
                    for result
                    in completed_results
                ]
            )
        )

        mean_steps = float(
            np.mean(
                [
                    result[
                        "steps"
                    ]
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
            len(
                completed_results
            ),

        "total_orders":
            len(
                results
            ),

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
    Compare two development checkpoints.

    Priority:
        1. Higher completion rate
        2. Higher collection fraction
        3. Lower optimality gap
    """

    if best_summary is None:
        return True

    # --------------------------------------------------------
    # Completion rate
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
        np.isnan(
            new_gap
        )
        and
        np.isnan(
            best_gap
        )
    ):
        return False

    if np.isnan(
        new_gap
    ):
        return False

    if np.isnan(
        best_gap
    ):
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
        "DQN ONE-PICK POTENTIAL-SHAPING TRAINING\n"
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
        f"Shaping scale:         {SHAPING_SCALE}\n"
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
    # HISTORY
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
    # INITIAL GREEDY EVALUATION
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

    # Save baseline checkpoint so best-model path always exists.
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
    # TRAINING
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
        # Store training result
        # ----------------------------------------------------

        training_row = {
            "episode":
                episode,

            "order":
                str(
                    sorted(
                        order
                    )
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
            or
            episode % 10 == 0
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
                f" | shaped "
                f"{metrics['reward']:8.1f}"
                f" | base "
                f"{metrics['base_reward']:7.1f}"
                f" | shaping "
                f"{metrics['shaping_reward']:7.1f}"
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
        # DEVELOPMENT EVALUATION
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
            # Save best development checkpoint
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
            # Intermediate result saving
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
    # TRAINING FINISHED
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
    # SAVE FINAL CSV OUTPUT
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
        "ONE-PICK POTENTIAL-SHAPING RUN COMPLETE\n"
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
    # BEST DEVELOPMENT RESULT
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

        if np.isfinite(
            gap
        ):

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
