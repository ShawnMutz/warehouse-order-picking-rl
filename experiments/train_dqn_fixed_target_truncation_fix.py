"""
Fixed-target one-pick DQN diagnostic.

Purpose
-------
Test whether the DQN can learn the simplest possible warehouse
routing problem:

    depot -> one fixed pick -> depot

Unlike the earlier one-pick experiments, the target location does
not change between episodes. This removes generalisation across the
96 pick-location bits as a source of difficulty.

Training:
    - fixed target: (8, 7)
    - potential-based reward shaping
    - original environment reward retained
    - 500 episodes
    - maximum 150 actions per episode

Evaluation:
    - greedy policy
    - same fixed target
    - no replay-buffer updates
    - no gradient updates
    - compared with exact optimal distance

Additional diagnostics:
    - items collected
    - completion
    - travel distance
    - invalid actions
    - invalid-action percentage
    - optimality gap

This is a diagnostic development experiment, not a final
experimental result.
"""

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
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RUN_NAME = "one_pick_fixed_target_truncation_fix"

TRAIN_SEED = 42

TRAIN_EPISODES = 500

MAX_STEPS = 150

EVALUATE_EVERY = 25


# ============================================================
# FIXED TARGET
# ============================================================

FIXED_TARGET = (8, 7)


# ============================================================
# DQN HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 1e-4

GAMMA = 0.99

EPSILON_START = 1.0

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.99995

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

    Picking aisles:
        columns 1,3,5,7,9,11,13,15

    Pick rows:
        rows 1..12

    Depot:
        (0,0)
    """

    height = 14
    width = 17

    # 1 = obstacle
    # 0 = walkable
    grid = np.ones(
        (height, width),
        dtype=np.int8,
    )

    # Cross aisles
    grid[0, :] = 0
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
    Create the warehouse environment.
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
    Calculate shortest-path distance from start to every
    reachable walkable warehouse location.
    """

    rows, cols = grid.shape

    if grid[start] != 0:
        raise ValueError(
            f"BFS start location is not walkable: {start}"
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
    Pre-compute shortest-path distance maps.
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
# STATE DECODING
# ============================================================

def decode_agent_position(
    state,
    grid,
):
    """
    Recover the discrete picker coordinate from the
    normalised first two state values.
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
    Decode remaining required picks from the 96 binary
    state indicators.
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
            "all_pick_locations."
        )

    remaining_locations = []

    for index, bit in enumerate(
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
    Calculate collection progress from the state vector.
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
# ACTION DIAGNOSTICS
# ============================================================

def calculate_action_metrics(
    steps,
    travel_distance,
):
    """
    Calculate valid and invalid actions.

    Every valid movement increases travel distance by exactly
    one. Invalid actions increase the decision-step counter
    without increasing travel distance.

    Therefore:

        invalid actions = steps - travel distance
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
        float(invalid_action_rate),
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
    Calculate routing potential.

    Before collection:

        Phi(s) =
        -(distance picker -> target
          + distance target -> depot)

    After collection:

        Phi(s) =
        -distance picker -> depot
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
    # Pick already collected
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
    # Fixed one-pick experiment
    # --------------------------------------------------------

    if len(remaining_picks) != 1:

        raise ValueError(
            "Fixed-target experiment expected zero or one "
            "remaining pick."
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

    remaining_distance = (
        distance_to_target
        + target_to_depot
    )

    return -float(
        remaining_distance
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
    Potential-based shaping:

        F(s,s') =
        scale * [gamma * Phi(s') - Phi(s)]
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
# DIRECT GREEDY ACTION
# ============================================================

def select_greedy_action(
    agent,
    state,
):
    """
    Select argmax Q(s,a) directly.

    This prevents evaluation from altering epsilon.
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
    Run one complete fixed-target training episode.
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
        # Environment
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
        # Shaping
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
        # Replay
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

            if not np.isfinite(loss):

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
    # Collection metrics
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

    # --------------------------------------------------------
    # Invalid-action metrics
    # --------------------------------------------------------

    (
        valid_actions,
        invalid_actions,
        invalid_action_rate,
    ) = calculate_action_metrics(
        env.steps,
        env.travel_distance,
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

        "epsilon":
            float(
                agent.epsilon
            ),

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
# FIXED-TARGET GREEDY EVALUATION
# ============================================================

def evaluate_fixed_target(
    env,
    agent,
    order,
    optimal_distance,
):
    """
    Evaluate the greedy DQN on the fixed target.
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

    # --------------------------------------------------------
    # Collection
    # --------------------------------------------------------

    (
        items_collected,
        items_remaining,
        collection_fraction,
    ) = get_collection_metrics(
        state,
        len(order),
    )

    # --------------------------------------------------------
    # Invalid actions
    # --------------------------------------------------------

    (
        valid_actions,
        invalid_actions,
        invalid_action_rate,
    ) = calculate_action_metrics(
        env.steps,
        env.travel_distance,
    )

    # --------------------------------------------------------
    # Optimality gap
    # --------------------------------------------------------

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
# BEST MODEL COMPARISON
# ============================================================

def evaluation_is_better(
    new_result,
    best_result,
):
    """
    Compare fixed-target checkpoints.

    Priority:
        1. completion
        2. collection
        3. lower optimality gap if completed
        4. fewer invalid actions
    """

    if best_result is None:
        return True

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    new_complete = int(
        new_result[
            "completed"
        ]
    )

    best_complete = int(
        best_result[
            "completed"
        ]
    )

    if new_complete > best_complete:
        return True

    if new_complete < best_complete:
        return False

    # --------------------------------------------------------
    # Collection
    # --------------------------------------------------------

    new_collection = new_result[
        "collection_fraction"
    ]

    best_collection = best_result[
        "collection_fraction"
    ]

    if new_collection > best_collection:
        return True

    if new_collection < best_collection:
        return False

    # --------------------------------------------------------
    # Completed route quality
    # --------------------------------------------------------

    if new_result["completed"]:

        new_gap = new_result[
            "optimality_gap"
        ]

        best_gap = best_result[
            "optimality_gap"
        ]

        if np.isfinite(
            new_gap
        ):

            if not np.isfinite(
                best_gap
            ):
                return True

            if new_gap < best_gap:
                return True

            if new_gap > best_gap:
                return False

    # --------------------------------------------------------
    # Invalid actions
    # --------------------------------------------------------

    return (
        new_result[
            "invalid_action_rate"
        ]
        <
        best_result[
            "invalid_action_rate"
        ]
    )


# ============================================================
# CSV SAVER
# ============================================================

def save_csv(
    rows,
    output_path,
):
    """
    Save list-of-dictionaries to CSV.
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
        "============================================\n"
        "DQN FIXED-TARGET LEARNABILITY DIAGNOSTIC\n"
        "============================================"
    )

    print(
        f"Run name:          {RUN_NAME}\n"
        f"Fixed target:      {FIXED_TARGET}\n"
        f"Training episodes: {TRAIN_EPISODES}\n"
        f"Max steps:         {MAX_STEPS}\n"
        f"Evaluate every:    {EVALUATE_EVERY}\n"
        f"Training seed:     {TRAIN_SEED}\n"
        f"Epsilon decay:     {EPSILON_DECAY}\n"
        f"Shaping scale:     {SHAPING_SCALE}\n"
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

    if FIXED_TARGET not in all_pick_locations:

        raise ValueError(
            f"FIXED_TARGET {FIXED_TARGET} is not a valid "
            "warehouse pick location."
        )

    print(
        f"State dimension: "
        f"{2 + len(all_pick_locations)}"
    )

    print(
        f"Fixed pick locations: "
        f"{len(all_pick_locations)}"
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
    # EXACT OPTIMUM
    # ========================================================

    fixed_order = [
        FIXED_TARGET
    ]

    optimal_distance = (
        exact_optimal_distance(
            grid,
            depot,
            fixed_order,
        )
    )

    print(
        f"\nExact optimum for "
        f"{FIXED_TARGET}: "
        f"{optimal_distance:.0f}"
    )

    # ========================================================
    # ENVIRONMENT AND AGENT
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

    evaluation_history_path = (
        results_directory
        / "dqn_evaluation_history.csv"
    )

    best_model_path = (
        results_directory
        / "dqn_best_model.pt"
    )

    final_model_path = (
        results_directory
        / "dqn_final_model.pt"
    )

    # ========================================================
    # HISTORY
    # ========================================================

    training_history = []

    evaluation_history = []

    rolling_completion = deque(
        maxlen=50
    )

    rolling_collection = deque(
        maxlen=50
    )

    rolling_invalid_rate = deque(
        maxlen=50
    )

    best_evaluation = None

    # ========================================================
    # INITIAL GREEDY EVALUATION
    # ========================================================

    print(
        "\nEvaluating untrained policy..."
    )

    initial_evaluation = (
        evaluate_fixed_target(
            env,
            agent,
            fixed_order,
            optimal_distance,
        )
    )

    initial_row = {
        "episode": 0,
        **initial_evaluation,
    }

    evaluation_history.append(
        initial_row
    )

    best_evaluation = (
        initial_evaluation.copy()
    )

    agent.save(
        best_model_path
    )

    print(
        f"Episode 0"
        f" | collected "
        f"{initial_evaluation['items_collected']}/1"
        f" | complete "
        f"{int(initial_evaluation['completed'])}"
        f" | distance "
        f"{initial_evaluation['distance']:.0f}"
        f" | invalid "
        f"{initial_evaluation['invalid_actions']}"
        f"/{initial_evaluation['steps']}"
        f" "
        f"({initial_evaluation['invalid_action_rate']:.1f}%)"
    )

    # ========================================================
    # TRAINING
    # ========================================================

    training_start = (
        time.perf_counter()
    )

    for episode in range(
        1,
        TRAIN_EPISODES + 1,
    ):

        # Same target every episode.
        order = [
            FIXED_TARGET
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

        rolling_completion_rate = (
            100.0
            * float(
                np.mean(
                    rolling_completion
                )
            )
        )

        rolling_collection_rate = float(
            np.mean(
                rolling_collection
            )
        )

        rolling_mean_invalid_rate = float(
            np.mean(
                rolling_invalid_rate
            )
        )

        # ----------------------------------------------------
        # Training row
        # ----------------------------------------------------

        training_row = {
            "episode":
                episode,

            "target":
                str(
                    FIXED_TARGET
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

            "rolling_50_completion_rate":
                rolling_completion_rate,

            "rolling_50_collection_fraction":
                rolling_collection_rate,

            "rolling_50_invalid_action_rate":
                rolling_mean_invalid_rate,
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
                f"{episode:03d}/{TRAIN_EPISODES}"
                f" | collected "
                f"{metrics['items_collected']}/1"
                f" | complete "
                f"{int(metrics['completed'])}"
                f" | steps "
                f"{metrics['steps']:3d}"
                f" | distance "
                f"{metrics['distance']:3.0f}"
                f" | invalid "
                f"{metrics['invalid_action_rate']:5.1f}%"
                f" | roll complete "
                f"{rolling_completion_rate:5.1f}%"
                f" | roll invalid "
                f"{rolling_mean_invalid_rate:5.1f}%"
                f" | eps "
                f"{metrics['epsilon']:.3f}"
                f" | loss "
                f"{loss_text}"
            )

        # ====================================================
        # GREEDY EVALUATION
        # ====================================================

        if (
            episode
            % EVALUATE_EVERY
            == 0
        ):

            evaluation = (
                evaluate_fixed_target(
                    env,
                    agent,
                    fixed_order,
                    optimal_distance,
                )
            )

            evaluation_row = {
                "episode":
                    episode,

                **evaluation,
            }

            evaluation_history.append(
                evaluation_row
            )

            if np.isfinite(
                evaluation[
                    "optimality_gap"
                ]
            ):

                gap_text = (
                    f"{evaluation['optimality_gap']:.2f}%"
                )

            else:

                gap_text = "N/A"

            print(
                "\n"
                "--------------------------------------------"
            )

            print(
                f"GREEDY FIXED-TARGET EVALUATION "
                f"AT EPISODE {episode}"
            )

            print(
                "--------------------------------------------"
            )

            print(
                f"Target:              "
                f"{FIXED_TARGET}"
            )

            print(
                f"Collected:           "
                f"{evaluation['items_collected']}/1"
            )

            print(
                f"Completed:           "
                f"{evaluation['completed']}"
            )

            print(
                f"Travel distance:     "
                f"{evaluation['distance']:.0f}"
            )

            print(
                f"Optimal distance:    "
                f"{evaluation['optimal_distance']:.0f}"
            )

            print(
                f"Optimality gap:      "
                f"{gap_text}"
            )

            print(
                f"Steps:               "
                f"{evaluation['steps']}"
            )

            print(
                f"Invalid actions:     "
                f"{evaluation['invalid_actions']}"
                f"/{evaluation['steps']}"
            )

            print(
                f"Invalid-action rate: "
                f"{evaluation['invalid_action_rate']:.1f}%"
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if evaluation_is_better(
                evaluation,
                best_evaluation,
            ):

                best_evaluation = (
                    evaluation.copy()
                )

                best_evaluation[
                    "episode"
                ] = episode

                agent.save(
                    best_model_path
                )

                print(
                    "New best fixed-target model saved."
                )

            print()

            # ------------------------------------------------
            # Intermediate CSV save
            # ------------------------------------------------

            save_csv(
                training_history,
                training_history_path,
            )

            save_csv(
                evaluation_history,
                evaluation_history_path,
            )

    # ========================================================
    # FINISH
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
        evaluation_history,
        evaluation_history_path,
    )

    # ========================================================
    # FINAL GREEDY EVALUATION
    # ========================================================

    final_evaluation = (
        evaluate_fixed_target(
            env,
            agent,
            fixed_order,
            optimal_distance,
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        "============================================\n"
        "FIXED-TARGET DIAGNOSTIC COMPLETE\n"
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
        f"Final rolling training completion: "
        f"{100.0 * np.mean(rolling_completion):.1f}%"
    )

    print(
        f"Final rolling invalid-action rate: "
        f"{np.mean(rolling_invalid_rate):.1f}%"
    )

    print(
        "\nFinal greedy policy:"
    )

    print(
        f"  Collected: "
        f"{final_evaluation['items_collected']}/1"
    )

    print(
        f"  Completed: "
        f"{final_evaluation['completed']}"
    )

    print(
        f"  Distance: "
        f"{final_evaluation['distance']:.0f}"
    )

    print(
        f"  Optimal distance: "
        f"{optimal_distance:.0f}"
    )

    print(
        f"  Invalid actions: "
        f"{final_evaluation['invalid_actions']}"
        f"/{final_evaluation['steps']}"
    )

    print(
        f"  Invalid-action rate: "
        f"{final_evaluation['invalid_action_rate']:.1f}%"
    )

    if final_evaluation[
        "completed"
    ]:

        print(
            f"  Optimality gap: "
            f"{final_evaluation['optimality_gap']:.2f}%"
        )

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Training history:\n"
        f"    {training_history_path}"
    )

    print(
        f"  Evaluation history:\n"
        f"    {evaluation_history_path}"
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
