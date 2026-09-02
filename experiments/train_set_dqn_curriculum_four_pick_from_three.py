"""
Set-DQN curriculum experiment: three picks -> four picks.

Purpose
-------
Test whether knowledge learned by the successful three-pick Set-DQN
can support learning and generalisation on four-pick orders.

Source model
------------
Best checkpoint from:

    set_dqn_balanced_three_pick_20_cycles

The source model achieved strong unseen three-pick completion.

Only the learned policy-network weights are transferred.

The following are deliberately fresh:

    - replay buffer
    - optimiser state
    - training-step counter
    - epsilon schedule

Epsilon is reset to 1.0.

This prevents the curriculum experiment from inheriting replay data,
optimizer momentum, or a low-exploration state from the previous run.

Training design
---------------
Warehouse:
    96 physical pick locations.

Each cycle:
    96 four-pick orders.

Four compatible permutations of all 96 physical locations are aligned:

    order_i = [
        permutation_1[i],
        permutation_2[i],
        permutation_3[i],
        permutation_4[i],
    ]

Therefore each cycle contains:

    96 orders x 4 picks = 384 pick appearances

Every physical location appears exactly:

    4 times per cycle.

With 20 cycles:

    1920 training episodes
    80 appearances per physical location.

Development evaluation
----------------------
40 deterministic unseen four-pick combinations.

The exact four-pick combinations are excluded from training while
their individual locations remain represented in other orders.

The transferred three-pick policy is evaluated on the four-pick probe
set BEFORE any four-pick training.

State representation
--------------------
Set-DQN consumes the raw 98-dimensional WarehouseEnv state directly.

No CoordinateStateEncoder is used.

Reward shaping
--------------
Uses the existing multi-pick potential-based shaping implementation.

Exact optimisation is used only for development evaluation.
"""

from utils.reward_shaping import (
    build_distance_lookup,
    calculate_potential_shaping_reward,
    get_remaining_pick_locations,
)
from utils.order_generation import generate_order
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
# RUN
# ============================================================

RUN_NAME = "set_dqn_curriculum_four_pick_from_three_20_cycles"


# ============================================================
# SOURCE CHECKPOINT
# ============================================================

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "set_dqn_balanced_three_pick_20_cycles"
    / "set_dqn_best_model.pt"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TRAIN_SEED = 42

TRAIN_CYCLES = 20

ORDER_SIZE = 4

MAX_ORDER_SIZE = 20

MAX_STEPS = 150


# ============================================================
# DEVELOPMENT PROBES
# ============================================================

PROBE_ORDERS = 40

# New deterministic seed range dedicated to the four-pick
# development diagnostic.
PROBE_SEED_START = 85_000

EVALUATE_EVERY_CYCLES = 1


# ============================================================
# TRAINING METRICS
# ============================================================

ROLLING_WINDOW = 96

PRINT_EVERY = 48


# ============================================================
# DQN HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 1e-4

GAMMA = 0.99

# Deliberately reset exploration after weight transfer.
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

DEVICE = "cpu"


# ============================================================
# WAREHOUSE
# ============================================================

def create_warehouse():
    """
    Create the fixed 14 x 17 warehouse.
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
    Create the fixed WarehouseEnv.
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
# FRESH AGENT
# ============================================================

def create_fresh_agent(
    env,
):
    """
    Create a completely fresh Set-DQN.

    This gives us:
        - fresh optimiser
        - fresh replay buffer
        - epsilon = 1.0
        - training counter = 0

    The three-pick policy weights are loaded separately.
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
# POLICY-ONLY TRANSFER
# ============================================================

def extract_policy_state_dict(
    checkpoint,
):
    """
    Extract policy-network weights from a saved checkpoint.

    This deliberately avoids loading:
        - optimiser state
        - epsilon
        - replay state
        - training counters

    Several common checkpoint key names are supported so this
    remains compatible with the project's save implementation.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Unexpected checkpoint format. Expected a dictionary."
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

    # --------------------------------------------------------
    # Some save functions store the PyTorch state_dict itself.
    # Detect that case.
    # --------------------------------------------------------

    values = list(
        checkpoint.values()
    )

    if (
        values
        and all(
            torch.is_tensor(
                value
            )
            for value in values
        )
    ):

        return checkpoint

    raise RuntimeError(
        "Could not locate policy-network weights in the "
        "three-pick checkpoint.\n"
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def load_three_pick_policy_weights(
    agent,
    checkpoint_path,
):
    """
    Transfer ONLY the three-pick policy network parameters into
    the fresh four-pick Set-DQN.

    Because Set-DQN is cardinality-independent, the same network
    architecture can consume four active picks without changing
    input dimensionality.
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Three-pick source checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print(
        "\nLoading three-pick policy weights:"
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

    policy_state_dict = (
        extract_policy_state_dict(
            checkpoint
        )
    )

    agent.policy_net.load_state_dict(
        policy_state_dict
    )

    # --------------------------------------------------------
    # Target begins identical to transferred policy.
    # --------------------------------------------------------

    agent.target_net.load_state_dict(
        agent.policy_net.state_dict()
    )

    agent.target_net.eval()

    # --------------------------------------------------------
    # Explicitly retain fresh curriculum settings.
    # --------------------------------------------------------

    agent.epsilon = (
        EPSILON_START
    )

    if hasattr(
        agent,
        "training_steps",
    ):

        agent.training_steps = 0

    if hasattr(
        agent,
        "steps_done",
    ):

        agent.steps_done = 0

    print(
        "Three-pick policy weights transferred successfully."
    )

    print(
        "Fresh four-pick optimiser retained."
    )

    print(
        "Fresh replay buffer retained."
    )

    print(
        f"Epsilon reset to {agent.epsilon:.2f}."
    )


# ============================================================
# ORDER KEY
# ============================================================

def order_key(
    order,
):
    """
    Create order-independent key for four locations.
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
    Generate 40 deterministic unique four-pick development
    combinations.
    """

    probe_orders = []

    seen_keys = set()

    seed = (
        PROBE_SEED_START
    )

    attempts = 0

    max_attempts = 100_000

    while len(
        probe_orders
    ) < PROBE_ORDERS:

        attempts += 1

        if attempts > max_attempts:

            raise RuntimeError(
                "Unable to generate enough unique "
                "four-pick development probes."
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
# BALANCED FOUR-PICK CYCLE
# ============================================================

def build_balanced_cycle_orders(
    all_pick_locations,
    cycle_rng,
    forbidden_order_keys,
    used_training_order_keys,
):
    """
    Construct one balanced four-pick cycle.

    Four compatible permutations of all 96 physical locations
    are aligned.

    Each location therefore occurs exactly four times per cycle.

    Constraints:
        - four unique physical locations/order
        - no development combination
        - no duplicate combination within cycle
        - no duplicate combination across previous cycles
    """

    locations = [
        tuple(
            location
        )
        for location in all_pick_locations
    ]

    location_count = len(
        locations
    )

    max_cycle_attempts = 2_000

    max_permutation_attempts = 20_000

    for _ in range(
        max_cycle_attempts
    ):

        permutations = []

        construction_failed = False

        # ----------------------------------------------------
        # Build four compatible permutations.
        # ----------------------------------------------------

        for slot_index in range(
            ORDER_SIZE
        ):

            valid_permutation = None

            for _ in range(
                max_permutation_attempts
            ):

                candidate = list(
                    locations
                )

                cycle_rng.shuffle(
                    candidate
                )

                valid = True

                for position in range(
                    location_count
                ):

                    candidate_location = (
                        candidate[
                            position
                        ]
                    )

                    previous_locations = {
                        permutations[
                            previous_slot
                        ][
                            position
                        ]
                        for previous_slot
                        in range(
                            slot_index
                        )
                    }

                    if (
                        candidate_location
                        in previous_locations
                    ):

                        valid = False
                        break

                if valid:

                    valid_permutation = (
                        candidate
                    )

                    break

            if valid_permutation is None:

                construction_failed = True

                break

            permutations.append(
                valid_permutation
            )

        if construction_failed:

            continue

        # ----------------------------------------------------
        # Align the four permutations.
        # ----------------------------------------------------

        candidate_orders = []

        candidate_keys = set()

        cycle_valid = True

        for position in range(
            location_count
        ):

            order = [
                permutations[
                    slot_index
                ][
                    position
                ]
                for slot_index
                in range(
                    ORDER_SIZE
                )
            ]

            if len(
                set(
                    order
                )
            ) != ORDER_SIZE:

                cycle_valid = False

                break

            key = order_key(
                order
            )

            if key in forbidden_order_keys:

                cycle_valid = False

                break

            if key in candidate_keys:

                cycle_valid = False

                break

            if key in used_training_order_keys:

                cycle_valid = False

                break

            candidate_keys.add(
                key
            )

            candidate_orders.append(
                list(
                    key
                )
            )

        if (
            cycle_valid
            and len(
                candidate_orders
            ) == location_count
        ):

            return (
                candidate_orders,
                candidate_keys,
            )

    raise RuntimeError(
        "Unable to construct a valid balanced "
        "four-pick training cycle."
    )


# ============================================================
# COMPLETE TRAINING SCHEDULE
# ============================================================

def build_balanced_training_schedule(
    all_pick_locations,
    forbidden_order_keys,
):
    """
    Build all 20 balanced four-pick training cycles.
    """

    schedule = []

    used_training_order_keys = set()

    episode = 0

    for cycle_index in range(
        TRAIN_CYCLES
    ):

        cycle_number = (
            cycle_index
            + 1
        )

        cycle_rng = random.Random(
            TRAIN_SEED
            + 40_000
            + cycle_index
        )

        (
            cycle_orders,
            cycle_keys,
        ) = build_balanced_cycle_orders(
            all_pick_locations,
            cycle_rng,
            forbidden_order_keys,
            used_training_order_keys,
        )

        used_training_order_keys.update(
            cycle_keys
        )

        for (
            cycle_position,
            order,
        ) in enumerate(
            cycle_orders,
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
# SCHEDULE VALIDATION
# ============================================================

def validate_balanced_training_schedule(
    schedule,
    all_pick_locations,
    forbidden_order_keys,
):
    """
    Validate all balancing and leakage constraints.
    """

    location_count = len(
        all_pick_locations
    )

    orders_per_cycle = (
        location_count
    )

    expected_episodes = (
        orders_per_cycle
        * TRAIN_CYCLES
    )

    expected_total_exposure = (
        ORDER_SIZE
        * TRAIN_CYCLES
    )

    # --------------------------------------------------------
    # Overall episodes.
    # --------------------------------------------------------

    if len(
        schedule
    ) != expected_episodes:

        raise AssertionError(
            f"Schedule contains {len(schedule)} episodes; "
            f"expected {expected_episodes}."
        )

    # --------------------------------------------------------
    # Every exact training combination must be unique.
    # --------------------------------------------------------

    training_keys = [
        row[
            "order_key"
        ]
        for row in schedule
    ]

    if len(
        training_keys
    ) != len(
        set(
            training_keys
        )
    ):

        raise AssertionError(
            "Training schedule contains duplicate "
            "four-pick combinations."
        )

    # --------------------------------------------------------
    # Development leakage.
    # --------------------------------------------------------

    overlap = (
        set(
            training_keys
        )
        & forbidden_order_keys
    )

    if overlap:

        raise AssertionError(
            "Development combinations found in training: "
            f"{sorted(overlap)}"
        )

    # --------------------------------------------------------
    # Cycle-by-cycle balance.
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
                f"{len(cycle_rows)} orders; "
                f"expected {orders_per_cycle}."
            )

        cycle_locations = []

        for row in cycle_rows:

            order = row[
                "order"
            ]

            if len(
                order
            ) != ORDER_SIZE:

                raise AssertionError(
                    "Training order does not contain "
                    "four locations."
                )

            if len(
                set(
                    order
                )
            ) != ORDER_SIZE:

                raise AssertionError(
                    "Training order contains duplicate "
                    "physical locations."
                )

            cycle_locations.extend(
                order
            )

        cycle_counts = Counter(
            cycle_locations
        )

        if len(
            cycle_counts
        ) != location_count:

            raise AssertionError(
                f"Cycle {cycle_number} does not contain "
                "all 96 physical locations."
            )

        for location in all_pick_locations:

            if (
                cycle_counts[
                    location
                ]
                != ORDER_SIZE
            ):

                raise AssertionError(
                    f"Location {location} appears "
                    f"{cycle_counts[location]} times in "
                    f"cycle {cycle_number}; "
                    f"expected {ORDER_SIZE}."
                )

    # --------------------------------------------------------
    # Overall location exposure.
    # --------------------------------------------------------

    global_locations = [
        location
        for row in schedule
        for location in row[
            "order"
        ]
    ]

    global_counts = Counter(
        global_locations
    )

    for location in all_pick_locations:

        if (
            global_counts[
                location
            ]
            != expected_total_exposure
        ):

            raise AssertionError(
                f"Location {location} appears "
                f"{global_counts[location]} times overall; "
                f"expected {expected_total_exposure}."
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
    Calculate collection progress from raw state.
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
    Infer valid and invalid actions.
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
    Train one four-pick curriculum episode.
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

    total_training_reward = 0.0

    total_base_reward = 0.0

    total_shaping_reward = 0.0

    losses = []

    while not (
        terminated
        or truncated
    ):

        action = agent.select_action(
            environment_state
        )

        (
            next_environment_state,
            base_reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

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

        agent.remember(
            environment_state,
            action,
            training_reward,
            next_environment_state,
            terminated,
            truncated,
        )

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
    all_pick_locations,
):
    """
    Greedily evaluate one four-pick probe.
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
# EXACT OPTIMA
# ============================================================

def calculate_probe_optima(
    grid,
    depot,
    probe_orders,
):
    """
    Calculate exact optimal distances for the four-pick probes.
    """

    optima = []

    print(
        "\nCalculating exact four-pick probe distances..."
    )

    for (
        index,
        order,
    ) in enumerate(
        probe_orders,
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
    agent,
    probe_orders,
    probe_optima,
    all_pick_locations,
    episode,
    cycle,
):
    """
    Evaluate all unseen four-pick development probes.
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

                "probe_index":
                    probe_index,

                "order":
                    str(
                        order
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
    Checkpoint ranking:

        1. completion
        2. collection fraction
        3. optimality gap
        4. invalid-action rate
    """

    if best_summary is None:

        return True

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
        and not best_gap_finite
    ):

        return True

    if (
        best_gap_finite
        and not new_gap_finite
    ):

        return False

    if (
        new_gap_finite
        and best_gap_finite
    ):

        if new_gap < best_gap:

            return True

        if new_gap > best_gap:

            return False

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
# CSV
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
# CONFIG
# ============================================================

def save_experiment_config(
    output_path,
    total_training_episodes,
    exposure_per_location,
):
    """
    Save experiment configuration.
    """

    config = {
        "run_name":
            RUN_NAME,

        "agent":
            "SetDQNAgent",

        "initialisation":
            "three_pick_policy_weight_transfer",

        "source_model":
            str(
                SOURCE_MODEL_PATH
            ),

        "source_order_size":
            3,

        "target_order_size":
            ORDER_SIZE,

        "policy_weights_transferred":
            True,

        "optimizer_transferred":
            False,

        "replay_buffer_transferred":
            False,

        "epsilon_transferred":
            False,

        "training_counter_transferred":
            False,

        "epsilon_reset":
            EPSILON_START,

        "representation":
            (
                "permutation_invariant_shared_"
                "pick_encoder_mean_pooling"
            ),

        "training_seed":
            TRAIN_SEED,

        "training_cycles":
            TRAIN_CYCLES,

        "orders_per_cycle":
            96,

        "total_training_episodes":
            total_training_episodes,

        "location_exposure_per_cycle":
            ORDER_SIZE,

        "total_exposure_per_location":
            exposure_per_location,

        "development_probe_orders":
            PROBE_ORDERS,

        "probe_seed_start":
            PROBE_SEED_START,

        "probe_policy":
            "unseen_exact_four_pick_combinations",

        "max_steps":
            MAX_STEPS,

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
        "SET-DQN CURRICULUM: THREE PICKS -> FOUR PICKS\n"
        "================================================"
    )

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

    exposure_per_location = (
        ORDER_SIZE
        * TRAIN_CYCLES
    )

    evaluate_every = (
        orders_per_cycle
        * EVALUATE_EVERY_CYCLES
    )

    # ========================================================
    # FRESH ENVIRONMENT + AGENT
    # ========================================================

    env = create_environment(
        grid,
        depot,
        all_pick_locations,
    )

    agent = create_fresh_agent(
        env
    )

    # ========================================================
    # TRANSFER THREE-PICK POLICY WEIGHTS
    # ========================================================

    load_three_pick_policy_weights(
        agent,
        SOURCE_MODEL_PATH,
    )

    # ========================================================
    # BASIC FOUR-PICK NETWORK CHECK
    # ========================================================

    test_state = env.reset(
        all_pick_locations[
            :ORDER_SIZE
        ]
    )

    test_q_values = agent.get_q_values(
        test_state
    )

    if test_state.shape != (
        98,
    ):

        raise RuntimeError(
            f"Expected raw state shape (98,), "
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
        f"\nRun name:                 {RUN_NAME}\n"
        f"Source order size:        3\n"
        f"Target order size:        {ORDER_SIZE}\n"
        f"Training cycles:          {TRAIN_CYCLES}\n"
        f"Orders per cycle:         {orders_per_cycle}\n"
        f"Training episodes:        {total_training_episodes}\n"
        f"Exposure/location/cycle:  {ORDER_SIZE}\n"
        f"Exposure/location total:  {exposure_per_location}\n"
        f"Development probes:       {PROBE_ORDERS}\n"
        f"Probe seed start:         {PROBE_SEED_START}\n"
        f"Max episode steps:        {MAX_STEPS}\n"
        f"Epsilon reset:            {EPSILON_START}\n"
        f"Epsilon decay:            {EPSILON_DECAY}\n"
        f"Raw state dimension:      {test_state.shape[0]}\n"
        f"Pick embedding dim:       {PICK_EMBEDDING_DIM}\n"
        f"Device:                   {DEVICE}\n"
    )

    # ========================================================
    # DEVELOPMENT PROBES
    # ========================================================

    print(
        "Creating unseen four-pick development probes..."
    )

    probe_orders = create_probe_orders(
        all_pick_locations,
        aisle_columns,
    )

    probe_order_keys = {
        order_key(
            order
        )
        for order in probe_orders
    }

    if len(
        probe_order_keys
    ) != PROBE_ORDERS:

        raise AssertionError(
            "Development probe set contains duplicates."
        )

    print(
        f"Created {len(probe_orders)} "
        "unique four-pick development combinations."
    )

    # ========================================================
    # TRAINING SCHEDULE
    # ========================================================

    print(
        "\nBuilding balanced four-pick training schedule..."
    )

    training_schedule = (
        build_balanced_training_schedule(
            all_pick_locations,
            probe_order_keys,
        )
    )

    validate_balanced_training_schedule(
        training_schedule,
        all_pick_locations,
        probe_order_keys,
    )

    print(
        "Balanced schedule verified:"
    )

    print(
        f"  {orders_per_cycle} orders/cycle"
    )

    print(
        f"  {TRAIN_CYCLES} cycles"
    )

    print(
        f"  {total_training_episodes} episodes"
    )

    print(
        f"  {ORDER_SIZE} appearances/location/cycle"
    )

    print(
        f"  {exposure_per_location} "
        "appearances/location overall"
    )

    print(
        f"  {len(training_schedule)} "
        "unique training combinations"
    )

    print(
        "  0 development combinations in training"
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
    # EXACT PROBE OPTIMA
    # ========================================================

    probe_optima = calculate_probe_optima(
        grid,
        depot,
        probe_orders,
    )

    maximum_probe_optimum = max(
        probe_optima
    )

    print(
        f"\nMaximum development-probe optimum: "
        f"{maximum_probe_optimum:.0f}"
    )

    print(
        f"MAX_STEPS: {MAX_STEPS}"
    )

    if maximum_probe_optimum >= MAX_STEPS:

        raise RuntimeError(
            "A four-pick probe optimum reaches or exceeds "
            "MAX_STEPS."
        )

    # ========================================================
    # OUTPUTS
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
        / "set_dqn_training_history.csv"
    )

    evaluation_summary_path = (
        results_directory
        / "set_dqn_probe_summary.csv"
    )

    evaluation_raw_path = (
        results_directory
        / "set_dqn_probe_raw.csv"
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
    # SAVE CONFIG / DEFINITIONS
    # ========================================================

    save_experiment_config(
        config_path,
        total_training_episodes,
        exposure_per_location,
    )

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

                "order":
                    str(
                        row[
                            "order"
                        ]
                    ),
            }
            for row in training_schedule
        ],
        schedule_path,
    )

    save_csv(
        [
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
        ],
        probe_definition_path,
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
    # TRANSFER-ONLY EVALUATION
    # ========================================================

    print(
        "\nEvaluating transferred three-pick policy "
        "on four-pick probes BEFORE training..."
    )

    (
        initial_summary,
        initial_raw,
    ) = evaluate_probe_set(
        env,
        agent,
        probe_orders,
        probe_optima,
        all_pick_locations,
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
        f"Transferred policy"
        f" | completion "
        f"{initial_summary['completion_rate']:.1f}%"
        f" | any-pick "
        f"{initial_summary['any_pick_rate']:.1f}%"
        f" | items "
        f"{initial_summary['mean_items_collected']:.3f}/4"
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
        # History
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
            or episode % PRINT_EVERY == 0
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
                f" | collect "
                f"{metrics['items_collected']}/4"
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
        # DEVELOPMENT EVALUATION
        # ====================================================

        if (
            episode % evaluate_every
            == 0
        ):

            (
                evaluation_summary,
                evaluation_raw,
            ) = evaluate_probe_set(
                env,
                agent,
                probe_orders,
                probe_optima,
                all_pick_locations,
                episode=episode,
                cycle=cycle,
            )

            evaluation_summaries.append(
                evaluation_summary
            )

            evaluation_raw_results.extend(
                evaluation_raw
            )

            mean_gap = evaluation_summary[
                "mean_optimality_gap"
            ]

            median_gap = evaluation_summary[
                "median_optimality_gap"
            ]

            mean_gap_text = (
                f"{mean_gap:.2f}%"
                if np.isfinite(
                    mean_gap
                )
                else "N/A"
            )

            median_gap_text = (
                f"{median_gap:.2f}%"
                if np.isfinite(
                    median_gap
                )
                else "N/A"
            )

            print(
                "\n"
                "------------------------------------------------"
            )

            print(
                f"SET-DQN 3 -> 4 CURRICULUM EVALUATION "
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
                f"Completed probes:         "
                f"{evaluation_summary['completed_orders']}"
                f"/{evaluation_summary['total_orders']}"
            )

            print(
                f"Any-pick rate:            "
                f"{evaluation_summary['any_pick_rate']:.1f}%"
            )

            print(
                f"Mean items collected:     "
                f"{evaluation_summary['mean_items_collected']:.3f}/4"
            )

            print(
                f"Mean collection fraction: "
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
                    "New best four-pick curriculum model saved."
                )

            print()

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
    # COMPLETE
    # ========================================================

    training_time = (
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
        evaluation_summaries,
        evaluation_summary_path,
    )

    save_csv(
        evaluation_raw_results,
        evaluation_raw_path,
    )

    print(
        "\n"
        "================================================\n"
        "SET-DQN 3 -> 4 CURRICULUM COMPLETE\n"
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
        f"{exposure_per_location}"
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

    if best_evaluation_summary is not None:

        print(
            "\nBest four-pick curriculum checkpoint:"
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
            f"  Mean items: "
            f"{best_evaluation_summary['mean_items_collected']:.3f}/4"
        )

        print(
            f"  Collection fraction: "
            f"{best_evaluation_summary['mean_collection_fraction']:.3f}"
        )

        print(
            f"  Invalid-action rate: "
            f"{best_evaluation_summary['mean_invalid_action_rate']:.1f}%"
        )

        best_gap = best_evaluation_summary[
            "mean_optimality_gap"
        ]

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
        f"  Probe definitions:\n"
        f"    {probe_definition_path}"
    )

    print(
        f"  Training history:\n"
        f"    {training_history_path}"
    )

    print(
        f"  Probe summary:\n"
        f"    {evaluation_summary_path}"
    )

    print(
        f"  Probe raw:\n"
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
