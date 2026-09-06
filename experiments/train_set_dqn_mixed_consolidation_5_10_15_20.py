"""
Set-DQN mixed-size consolidation: 5 / 10 / 15 / 20 picks.

Purpose
-------
Consolidate the successful twenty-pick Set-DQN across the four order
sizes used in the final methodology after the retention diagnostic
showed weaker performance at 5 and 10 picks.

The model is initialised from the best checkpoint produced by:

    set_dqn_curriculum_twenty_pick_from_fourteen_8_cycles

Only policy-network weights are transferred. Optimiser state, replay
buffer, epsilon and training counters remain fresh.

Mixed training design
---------------------
Every cycle contains:

    96 five-pick orders
    96 ten-pick orders
    96 fifteen-pick orders
    96 twenty-pick orders

The 384 orders are randomly interleaved within each cycle.

For a given order size k, a shuffled circular ordering of all 96
locations is used to construct 96 cyclic windows of length k. This
guarantees:

    - k distinct locations per order
    - 96 orders for that cardinality per cycle
    - exactly k appearances of every location for that cardinality

Across 5, 10, 15 and 20 picks, every physical location therefore
appears:

    5 + 10 + 15 + 20 = 50 times per mixed cycle.

With 3 mixed cycles:

    384 x 3 = 1152 training episodes
    50 x 3 = 150 appearances per physical location

Development evaluation
----------------------
Fifty fresh deterministic uniform probes are used for each of the four
order sizes:

    5 picks  -> seeds from 450_000
    10 picks -> seeds from 460_000
    15 picks -> seeds from 470_000
    20 picks -> seeds from 480_000

The transferred twenty-pick policy is evaluated before training and
after every complete mixed cycle.

Checkpoint selection
--------------------
The best checkpoint is selected by:

    1. highest minimum completion across 5/10/15/20
    2. highest mean completion across the four sizes
    3. highest mean collection fraction
    4. lowest mean invalid-action rate

Episode 0 is a valid checkpoint candidate.

Exact optimisation is deliberately omitted in this consolidation stage.
Its purpose is retention/generalisation rather than route-quality
measurement. Exact benchmarking is reserved for the final experiment.

Episode horizon
---------------
MAX_STEPS is fixed at 250 for every order size, following the paired
fifteen-pick sensitivity analysis showing that the earlier 150-step
limit materially censored large-order completion.
"""

from utils.reward_shaping import (
    build_distance_lookup,
    calculate_potential_shaping_reward,
    get_remaining_pick_locations,
)
from utils.order_generation import generate_order
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
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RUN_NAME = "set_dqn_mixed_consolidation_5_10_15_20_3_cycles"

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "set_dqn_curriculum_twenty_pick_from_fourteen_8_cycles"
    / "set_dqn_best_model.pt"
)

TRAIN_SEED = 42

ORDER_SIZES = [
    5,
    10,
    15,
    20,
]

TRAIN_CYCLES = 3

ORDERS_PER_SIZE_PER_CYCLE = 96

MAX_ORDER_SIZE = 20

MAX_STEPS = 250


# ============================================================
# DEVELOPMENT PROBES
# ============================================================

PROBE_ORDERS_PER_SIZE = 50

PROBE_SEED_START_BY_SIZE = {
    5: 450_000,
    10: 460_000,
    15: 470_000,
    20: 480_000,
}

EVALUATE_EVERY_CYCLES = 1


# ============================================================
# TRAINING METRICS
# ============================================================

ORDERS_PER_MIXED_CYCLE = (
    len(ORDER_SIZES)
    * ORDERS_PER_SIZE_PER_CYCLE
)

ROLLING_WINDOW = ORDERS_PER_MIXED_CYCLE

PRINT_EVERY = 48


# ============================================================
# SET-DQN HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 1e-4

GAMMA = 0.99

EPSILON_START = 0.30

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
    """Create the fixed 14 x 17 warehouse."""

    height = 14
    width = 17

    grid = np.ones(
        (height, width),
        dtype=np.int8,
    )

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

    if len(all_pick_locations) != 96:
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

def set_random_seeds(seed):
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

def create_fresh_agent(env):
    """
    Create a fresh Set-DQN optimiser/replay/exploration state.

    Policy weights are transferred separately.
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

def extract_policy_state_dict(checkpoint):
    """Extract policy-network weights from a saved checkpoint."""

    if not isinstance(checkpoint, dict):
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
            candidate = checkpoint[key]

            if isinstance(candidate, dict):
                return candidate

    values = list(checkpoint.values())

    if values and all(
        torch.is_tensor(value)
        for value in values
    ):
        return checkpoint

    raise RuntimeError(
        "Could not locate policy-network weights in the "
        "twenty-pick checkpoint.\n"
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def load_twenty_pick_policy_weights(
    agent,
    checkpoint_path,
):
    """
    Transfer only the successful twenty-pick policy parameters.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Twenty-pick source checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print(
        "\nLoading twenty-pick policy weights:"
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

    policy_state_dict = extract_policy_state_dict(
        checkpoint
    )

    agent.policy_net.load_state_dict(
        policy_state_dict
    )

    agent.target_net.load_state_dict(
        agent.policy_net.state_dict()
    )

    agent.target_net.eval()

    agent.epsilon = EPSILON_START

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
        "Twenty-pick policy weights transferred successfully."
    )

    print(
        "Fresh mixed-size optimiser retained."
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

def order_key(order):
    return tuple(
        sorted(
            tuple(location)
            for location in order
        )
    )


# ============================================================
# DEVELOPMENT PROBES
# ============================================================

def create_probe_orders_for_size(
    all_pick_locations,
    aisle_columns,
    order_size,
):
    """Generate fresh deterministic uniform probes for one size."""

    if order_size not in PROBE_SEED_START_BY_SIZE:
        raise ValueError(
            f"No probe seed configured for size {order_size}."
        )

    rows = []

    seen_keys = set()

    seed = PROBE_SEED_START_BY_SIZE[
        order_size
    ]

    attempts = 0

    max_attempts = 100_000

    while len(rows) < PROBE_ORDERS_PER_SIZE:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Unable to generate enough unique "
                f"{order_size}-pick probes."
            )

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=order_size,
            distribution="uniform",
            seed=seed,
            aisle_columns=aisle_columns,
        )

        key = order_key(
            order
        )

        if key not in seen_keys:
            rows.append(
                {
                    "order_size":
                        order_size,

                    "probe_index":
                        len(rows) + 1,

                    "seed":
                        seed,

                    "order":
                        list(key),

                    "order_key":
                        key,
                }
            )

            seen_keys.add(
                key
            )

        seed += 1

    return rows


def create_all_probe_orders(
    all_pick_locations,
    aisle_columns,
):
    """Create 50 fresh probes for every mixed training size."""

    probe_rows_by_size = {}

    for order_size in ORDER_SIZES:
        probe_rows_by_size[
            order_size
        ] = create_probe_orders_for_size(
            all_pick_locations,
            aisle_columns,
            order_size,
        )

    return probe_rows_by_size


# ============================================================
# BALANCED ORDERS FOR ONE SIZE / ONE CYCLE
# ============================================================

def build_balanced_orders_for_size(
    all_pick_locations,
    order_size,
    cycle_rng,
    forbidden_order_keys,
    used_training_order_keys,
):
    """
    Build 96 balanced orders of one cardinality.

    A shuffled circular ordering of all locations is created and
    windows of length `order_size` are taken from all 96 starting
    positions.
    """

    locations = [
        tuple(location)
        for location in all_pick_locations
    ]

    location_count = len(
        locations
    )

    if order_size > location_count:
        raise ValueError(
            "Order size cannot exceed number of pick locations."
        )

    max_cycle_attempts = 10_000

    for _ in range(
        max_cycle_attempts
    ):
        base_permutation = list(
            locations
        )

        cycle_rng.shuffle(
            base_permutation
        )

        candidate_orders = []

        candidate_keys = set()

        cycle_valid = True

        for position in range(
            location_count
        ):
            order = [
                base_permutation[
                    (
                        position
                        + slot_index
                    )
                    % location_count
                ]
                for slot_index in range(
                    order_size
                )
            ]

            if len(
                set(order)
            ) != order_size:
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
                list(key)
            )

        if (
            cycle_valid
            and len(candidate_orders)
            == location_count
        ):
            return (
                candidate_orders,
                candidate_keys,
            )

    raise RuntimeError(
        f"Unable to construct balanced "
        f"{order_size}-pick training orders."
    )


# ============================================================
# COMPLETE MIXED TRAINING SCHEDULE
# ============================================================

def build_mixed_training_schedule(
    all_pick_locations,
    forbidden_keys_by_size,
):
    """
    Build three mixed cycles with 96 orders at each cardinality.

    Each cycle's 384 orders are shuffled after the four balanced
    cardinality-specific batches have been constructed.
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

        mixed_cycle_rows = []

        for order_size in ORDER_SIZES:
            size_rng = random.Random(
                TRAIN_SEED
                + 100_000
                + cycle_index * 10_000
                + order_size * 100
            )

            (
                size_orders,
                size_keys,
            ) = build_balanced_orders_for_size(
                all_pick_locations,
                order_size,
                size_rng,
                forbidden_keys_by_size[
                    order_size
                ],
                used_training_order_keys,
            )

            used_training_order_keys.update(
                size_keys
            )

            for within_size_position, order in enumerate(
                size_orders,
                start=1,
            ):
                mixed_cycle_rows.append(
                    {
                        "cycle":
                            cycle_number,

                        "order_size":
                            order_size,

                        "within_size_position":
                            within_size_position,

                        "order":
                            order,

                        "order_key":
                            order_key(order),
                    }
                )

        mix_rng = random.Random(
            TRAIN_SEED
            + 900_000
            + cycle_index
        )

        mix_rng.shuffle(
            mixed_cycle_rows
        )

        for cycle_position, row in enumerate(
            mixed_cycle_rows,
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

                    **row,
                }
            )

    return schedule


# ============================================================
# SCHEDULE VALIDATION
# ============================================================

def validate_mixed_training_schedule(
    schedule,
    all_pick_locations,
    forbidden_keys_by_size,
):
    """Validate balance, uniqueness and probe exclusion."""

    location_count = len(
        all_pick_locations
    )

    expected_total_episodes = (
        ORDERS_PER_MIXED_CYCLE
        * TRAIN_CYCLES
    )

    expected_total_exposure = (
        sum(ORDER_SIZES)
        * TRAIN_CYCLES
    )

    if len(schedule) != expected_total_episodes:
        raise AssertionError(
            f"Schedule contains {len(schedule)} episodes; "
            f"expected {expected_total_episodes}."
        )

    training_keys = [
        row["order_key"]
        for row in schedule
    ]

    if len(training_keys) != len(
        set(training_keys)
    ):
        raise AssertionError(
            "Mixed training schedule contains duplicate combinations."
        )

    for order_size in ORDER_SIZES:
        training_size_keys = {
            row["order_key"]
            for row in schedule
            if row["order_size"] == order_size
        }

        overlap = (
            training_size_keys
            & forbidden_keys_by_size[
                order_size
            ]
        )

        if overlap:
            raise AssertionError(
                f"{order_size}-pick development probes "
                "found in training schedule."
            )

    for cycle_number in range(
        1,
        TRAIN_CYCLES + 1,
    ):
        cycle_rows = [
            row
            for row in schedule
            if row["cycle"]
            == cycle_number
        ]

        if len(cycle_rows) != ORDERS_PER_MIXED_CYCLE:
            raise AssertionError(
                f"Cycle {cycle_number} contains "
                f"{len(cycle_rows)} episodes; expected "
                f"{ORDERS_PER_MIXED_CYCLE}."
            )

        cycle_total_counts = Counter()

        for order_size in ORDER_SIZES:
            size_rows = [
                row
                for row in cycle_rows
                if row["order_size"]
                == order_size
            ]

            if len(size_rows) != ORDERS_PER_SIZE_PER_CYCLE:
                raise AssertionError(
                    f"Cycle {cycle_number}, size {order_size}: "
                    f"{len(size_rows)} orders; expected "
                    f"{ORDERS_PER_SIZE_PER_CYCLE}."
                )

            size_counts = Counter()

            for row in size_rows:
                order = row[
                    "order"
                ]

                if len(order) != order_size:
                    raise AssertionError(
                        f"Expected {order_size} picks in order."
                    )

                if len(
                    set(order)
                ) != order_size:
                    raise AssertionError(
                        "Training order contains duplicate locations."
                    )

                size_counts.update(
                    order
                )

                cycle_total_counts.update(
                    order
                )

            for location in all_pick_locations:
                if size_counts[
                    location
                ] != order_size:
                    raise AssertionError(
                        f"Cycle {cycle_number}, size {order_size}: "
                        f"location {location} appears "
                        f"{size_counts[location]} times; "
                        f"expected {order_size}."
                    )

        expected_cycle_exposure = sum(
            ORDER_SIZES
        )

        for location in all_pick_locations:
            if cycle_total_counts[
                location
            ] != expected_cycle_exposure:
                raise AssertionError(
                    f"Cycle {cycle_number}: location {location} "
                    f"appears {cycle_total_counts[location]} times; "
                    f"expected {expected_cycle_exposure}."
                )

    global_counts = Counter(
        location
        for row in schedule
        for location in row[
            "order"
        ]
    )

    for location in all_pick_locations:
        if global_counts[
            location
        ] != expected_total_exposure:
            raise AssertionError(
                f"Location {location} appears "
                f"{global_counts[location]} times overall; "
                f"expected {expected_total_exposure}."
            )


# ============================================================
# COLLECTION / ACTION METRICS
# ============================================================

def get_collection_metrics(
    environment_state,
    all_pick_locations,
    order_size,
):
    remaining_picks = get_remaining_pick_locations(
        environment_state,
        all_pick_locations,
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


def calculate_action_metrics(
    steps,
    travel_distance,
):
    valid_actions = int(
        round(
            travel_distance
        )
    )

    invalid_actions = max(
        0,
        int(steps)
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
    """Train Set-DQN on one variable-cardinality order."""

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
            float(base_reward)
            + float(shaping_reward)
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
        len(order),
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
            np.mean(losses)
        )
        if losses
        else np.nan
    )

    all_items_collected = (
        items_collected
        == len(order)
    )

    no_finish = (
        all_items_collected
        and not bool(
            terminated
        )
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

        "all_items_collected":
            bool(
                all_items_collected
            ),

        "collected_all_but_not_completed":
            bool(
                no_finish
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

def evaluate_order(
    env,
    agent,
    order,
    all_pick_locations,
):
    environment_state = env.reset(
        order
    )

    terminated = False

    truncated = False

    start_time = time.perf_counter()

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
        len(order),
    )

    (
        valid_actions,
        invalid_actions,
        invalid_action_rate,
    ) = calculate_action_metrics(
        env.steps,
        env.travel_distance,
    )

    all_items_collected = (
        items_collected
        == len(order)
    )

    no_finish = (
        all_items_collected
        and not bool(
            terminated
        )
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

        "all_items_collected":
            bool(
                all_items_collected
            ),

        "collected_all_but_not_completed":
            bool(
                no_finish
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
# MIXED DEVELOPMENT EVALUATION
# ============================================================

def evaluate_probe_sets(
    env,
    agent,
    probe_rows_by_size,
    all_pick_locations,
    episode,
    cycle,
):
    """
    Evaluate all 200 probes and return:
        - aggregate checkpoint summary
        - one per-size summary row per cardinality
        - raw per-order rows
    """

    raw_results = []

    per_size_summaries = []

    for order_size in ORDER_SIZES:
        size_raw = []

        for probe_row in probe_rows_by_size[
            order_size
        ]:
            evaluation = evaluate_order(
                env,
                agent,
                probe_row[
                    "order"
                ],
                all_pick_locations,
            )

            row = {
                "episode":
                    episode,

                "cycle":
                    cycle,

                "order_size":
                    order_size,

                "probe_index":
                    probe_row[
                        "probe_index"
                    ],

                "seed":
                    probe_row[
                        "seed"
                    ],

                "order":
                    str(
                        probe_row[
                            "order"
                        ]
                    ),

                **evaluation,
            }

            size_raw.append(
                row
            )

            raw_results.append(
                row
            )

        completed_count = sum(
            row[
                "completed"
            ]
            for row in size_raw
        )

        total_orders = len(
            size_raw
        )

        completion_rate = (
            100.0
            * completed_count
            / total_orders
        )

        any_pick_count = sum(
            row[
                "items_collected"
            ] > 0
            for row in size_raw
        )

        any_pick_rate = (
            100.0
            * any_pick_count
            / total_orders
        )

        all_items_collected_count = sum(
            row[
                "all_items_collected"
            ]
            for row in size_raw
        )

        all_items_collected_rate = (
            100.0
            * all_items_collected_count
            / total_orders
        )

        no_finish_count = sum(
            row[
                "collected_all_but_not_completed"
            ]
            for row in size_raw
        )

        no_finish_rate = (
            100.0
            * no_finish_count
            / total_orders
        )

        mean_items_collected = float(
            np.mean(
                [
                    row[
                        "items_collected"
                    ]
                    for row in size_raw
                ]
            )
        )

        mean_collection_fraction = float(
            np.mean(
                [
                    row[
                        "collection_fraction"
                    ]
                    for row in size_raw
                ]
            )
        )

        mean_invalid_action_rate = float(
            np.mean(
                [
                    row[
                        "invalid_action_rate"
                    ]
                    for row in size_raw
                ]
            )
        )

        completed_distances = [
            row[
                "distance"
            ]
            for row in size_raw
            if row[
                "completed"
            ]
        ]

        mean_completed_distance = (
            float(
                np.mean(
                    completed_distances
                )
            )
            if completed_distances
            else np.nan
        )

        per_size_summaries.append(
            {
                "episode":
                    episode,

                "cycle":
                    cycle,

                "order_size":
                    order_size,

                "completion_rate":
                    completion_rate,

                "completed_orders":
                    completed_count,

                "total_orders":
                    total_orders,

                "any_pick_rate":
                    any_pick_rate,

                "all_items_collected_rate":
                    all_items_collected_rate,

                "collected_all_but_not_completed_rate":
                    no_finish_rate,

                "mean_items_collected":
                    mean_items_collected,

                "mean_collection_fraction":
                    mean_collection_fraction,

                "mean_invalid_action_rate":
                    mean_invalid_action_rate,

                "mean_completed_distance":
                    mean_completed_distance,
            }
        )

    completion_rates = [
        row[
            "completion_rate"
        ]
        for row in per_size_summaries
    ]

    collection_fractions = [
        row[
            "mean_collection_fraction"
        ]
        for row in per_size_summaries
    ]

    invalid_rates = [
        row[
            "mean_invalid_action_rate"
        ]
        for row in per_size_summaries
    ]

    aggregate_summary = {
        "episode":
            episode,

        "cycle":
            cycle,

        "minimum_completion_rate":
            float(
                min(
                    completion_rates
                )
            ),

        "mean_completion_rate":
            float(
                np.mean(
                    completion_rates
                )
            ),

        "mean_collection_fraction":
            float(
                np.mean(
                    collection_fractions
                )
            ),

        "mean_invalid_action_rate":
            float(
                np.mean(
                    invalid_rates
                )
            ),
    }

    for size_summary in per_size_summaries:
        order_size = size_summary[
            "order_size"
        ]

        aggregate_summary[
            f"completion_{order_size}"
        ] = size_summary[
            "completion_rate"
        ]

        aggregate_summary[
            f"collection_{order_size}"
        ] = size_summary[
            "mean_collection_fraction"
        ]

        aggregate_summary[
            f"invalid_{order_size}"
        ] = size_summary[
            "mean_invalid_action_rate"
        ]

    return (
        aggregate_summary,
        per_size_summaries,
        raw_results,
    )


# ============================================================
# BEST CHECKPOINT
# ============================================================

def mixed_evaluation_is_better(
    new_summary,
    best_summary,
):
    """
    Maximin checkpoint ranking:
        1. higher minimum completion
        2. higher mean completion
        3. higher mean collection
        4. lower mean invalid-action rate
    """

    if best_summary is None:
        return True

    if (
        new_summary[
            "minimum_completion_rate"
        ]
        > best_summary[
            "minimum_completion_rate"
        ]
    ):
        return True

    if (
        new_summary[
            "minimum_completion_rate"
        ]
        < best_summary[
            "minimum_completion_rate"
        ]
    ):
        return False

    if (
        new_summary[
            "mean_completion_rate"
        ]
        > best_summary[
            "mean_completion_rate"
        ]
    ):
        return True

    if (
        new_summary[
            "mean_completion_rate"
        ]
        < best_summary[
            "mean_completion_rate"
        ]
    ):
        return False

    if (
        new_summary[
            "mean_collection_fraction"
        ]
        > best_summary[
            "mean_collection_fraction"
        ]
    ):
        return True

    if (
        new_summary[
            "mean_collection_fraction"
        ]
        < best_summary[
            "mean_collection_fraction"
        ]
    ):
        return False

    return (
        new_summary[
            "mean_invalid_action_rate"
        ]
        < best_summary[
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
    location_count,
    total_training_episodes,
    total_exposure_per_location,
):
    config = {
        "run_name":
            RUN_NAME,

        "agent":
            "SetDQNAgent",

        "initialisation":
            "twenty_pick_policy_weight_transfer",

        "training_type":
            "mixed_size_consolidation",

        "source_model":
            str(
                SOURCE_MODEL_PATH
            ),

        "source_order_size":
            20,

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

        "order_sizes":
            ORDER_SIZES,

        "training_cycles":
            TRAIN_CYCLES,

        "orders_per_size_per_cycle":
            ORDERS_PER_SIZE_PER_CYCLE,

        "orders_per_mixed_cycle":
            ORDERS_PER_MIXED_CYCLE,

        "total_training_episodes":
            total_training_episodes,

        "location_count":
            location_count,

        "location_exposure_per_cycle":
            sum(
                ORDER_SIZES
            ),

        "total_exposure_per_location":
            total_exposure_per_location,

        "mixed_order_interleaving":
            True,

        "balanced_schedule_method":
            "cardinality_specific_cyclic_windows_then_cycle_shuffle",

        "development_probe_orders_per_size":
            PROBE_ORDERS_PER_SIZE,

        "development_probe_total":
            (
                PROBE_ORDERS_PER_SIZE
                * len(
                    ORDER_SIZES
                )
            ),

        "probe_seed_start_by_size":
            PROBE_SEED_START_BY_SIZE,

        "checkpoint_ranking":
            [
                "highest_minimum_completion_across_sizes",
                "highest_mean_completion",
                "highest_mean_collection_fraction",
                "lowest_mean_invalid_action_rate",
            ],

        "episode_zero_checkpoint_candidate":
            True,

        "exact_benchmark":
            False,

        "max_steps":
            MAX_STEPS,

        "max_steps_rationale":
            (
                "250_steps_retained_after_paired_fifteen_pick_"
                "sensitivity_analysis_showed_150_steps_censored_"
                "large_order_completion"
            ),

        "representation":
            "permutation_invariant_set_dqn_raw_98_state",

        "max_order_size":
            MAX_ORDER_SIZE,

        "training_seed":
            TRAIN_SEED,

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
# EVALUATION PRINTING
# ============================================================

def print_mixed_evaluation(
    aggregate_summary,
    per_size_summaries,
    label,
):
    print(
        "\n"
        "------------------------------------------------"
    )

    print(
        label
    )

    print(
        "------------------------------------------------"
    )

    for summary in per_size_summaries:
        print(
            f"{summary['order_size']:2d} picks"
            f" | complete "
            f"{summary['completion_rate']:5.1f}%"
            f" | collection "
            f"{summary['mean_collection_fraction']:.3f}"
            f" | all-picked "
            f"{summary['all_items_collected_rate']:5.1f}%"
            f" | no-finish "
            f"{summary['collected_all_but_not_completed_rate']:5.1f}%"
            f" | invalid "
            f"{summary['mean_invalid_action_rate']:5.1f}%"
        )

    print(
        "\nCheckpoint metrics:"
    )

    print(
        f"  Minimum completion: "
        f"{aggregate_summary['minimum_completion_rate']:.1f}%"
    )

    print(
        f"  Mean completion:    "
        f"{aggregate_summary['mean_completion_rate']:.1f}%"
    )

    print(
        f"  Mean collection:    "
        f"{aggregate_summary['mean_collection_fraction']:.3f}"
    )

    print(
        f"  Mean invalid rate:  "
        f"{aggregate_summary['mean_invalid_action_rate']:.1f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "\n"
        "================================================\n"
        "SET-DQN MIXED CONSOLIDATION: 5 / 10 / 15 / 20\n"
        "================================================"
    )

    set_random_seeds(
        TRAIN_SEED
    )

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
        ORDERS_PER_MIXED_CYCLE
        * TRAIN_CYCLES
    )

    exposure_per_location_per_cycle = sum(
        ORDER_SIZES
    )

    total_exposure_per_location = (
        exposure_per_location_per_cycle
        * TRAIN_CYCLES
    )

    evaluate_every = (
        ORDERS_PER_MIXED_CYCLE
        * EVALUATE_EVERY_CYCLES
    )

    env = create_environment(
        grid,
        depot,
        all_pick_locations,
    )

    agent = create_fresh_agent(
        env
    )

    load_twenty_pick_policy_weights(
        agent,
        SOURCE_MODEL_PATH,
    )

    # Basic network check using the largest cardinality.
    test_state = env.reset(
        all_pick_locations[
            :MAX_ORDER_SIZE
        ]
    )

    test_q_values = agent.get_q_values(
        test_state
    )

    if test_state.shape != (
        98,
    ):
        raise RuntimeError(
            f"Expected raw state (98,), "
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
        f"\nRun name:                    {RUN_NAME}\n"
        f"Agent:                       SetDQN\n"
        f"Initialisation:              twenty-pick policy transfer\n"
        f"Training sizes:              {ORDER_SIZES}\n"
        f"Epsilon reset:               {EPSILON_START}\n"
        f"Mixed cycles:                {TRAIN_CYCLES}\n"
        f"Locations:                   {location_count}\n"
        f"Orders/size/cycle:           {ORDERS_PER_SIZE_PER_CYCLE}\n"
        f"Orders/mixed cycle:          {ORDERS_PER_MIXED_CYCLE}\n"
        f"Training episodes:           {total_training_episodes}\n"
        f"Exposure/location/cycle:     {exposure_per_location_per_cycle}\n"
        f"Exposure/location total:     {total_exposure_per_location}\n"
        f"Development probes/size:     {PROBE_ORDERS_PER_SIZE}\n"
        f"Total development probes:    "
        f"{PROBE_ORDERS_PER_SIZE * len(ORDER_SIZES)}\n"
        f"Probe seed starts:           {PROBE_SEED_START_BY_SIZE}\n"
        f"Evaluate every:              {evaluate_every} episodes\n"
        f"Max episode steps:           {MAX_STEPS}\n"
        f"Training seed:               {TRAIN_SEED}\n"
        f"Epsilon decay:               {EPSILON_DECAY}\n"
        f"Shaping scale:               {SHAPING_SCALE}\n"
        f"Raw state dimension:         {test_state.shape[0]}\n"
        f"Pick embedding dim:          {PICK_EMBEDDING_DIM}\n"
        f"Device:                      {DEVICE}\n"
    )

    # ========================================================
    # DEVELOPMENT PROBES
    # ========================================================

    print(
        "Creating fresh mixed-size development probes..."
    )

    probe_rows_by_size = create_all_probe_orders(
        all_pick_locations,
        aisle_columns,
    )

    forbidden_keys_by_size = {}

    for order_size in ORDER_SIZES:
        rows = probe_rows_by_size[
            order_size
        ]

        keys = {
            row[
                "order_key"
            ]
            for row in rows
        }

        if len(keys) != PROBE_ORDERS_PER_SIZE:
            raise AssertionError(
                f"{order_size}-pick probe set contains duplicates."
            )

        forbidden_keys_by_size[
            order_size
        ] = keys

        print(
            f"  {order_size:2d} picks: "
            f"{len(rows)} unique probes "
            f"from seed "
            f"{PROBE_SEED_START_BY_SIZE[order_size]}"
        )

    # ========================================================
    # TRAINING SCHEDULE
    # ========================================================

    print(
        "\nBuilding balanced mixed-size training schedule..."
    )

    training_schedule = build_mixed_training_schedule(
        all_pick_locations,
        forbidden_keys_by_size,
    )

    validate_mixed_training_schedule(
        training_schedule,
        all_pick_locations,
        forbidden_keys_by_size,
    )

    print(
        "Balanced mixed schedule verified:"
    )

    print(
        f"  {ORDERS_PER_SIZE_PER_CYCLE} orders/size/cycle"
    )

    print(
        f"  {ORDERS_PER_MIXED_CYCLE} episodes/mixed cycle"
    )

    print(
        f"  {TRAIN_CYCLES} mixed cycles"
    )

    print(
        f"  {total_training_episodes} total episodes"
    )

    print(
        f"  {exposure_per_location_per_cycle} "
        "appearances/location/cycle"
    )

    print(
        f"  {total_exposure_per_location} "
        "appearances/location total"
    )

    print(
        f"  {len(training_schedule)} "
        "unique training combinations"
    )

    print(
        "  0 development combinations in training"
    )

    # ========================================================
    # SHORTEST PATHS
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

    aggregate_summary_path = (
        results_directory
        / "set_dqn_mixed_summary.csv"
    )

    per_size_summary_path = (
        results_directory
        / "set_dqn_mixed_per_size_summary.csv"
    )

    evaluation_raw_path = (
        results_directory
        / "set_dqn_mixed_raw.csv"
    )

    best_model_path = (
        results_directory
        / "set_dqn_best_model.pt"
    )

    final_model_path = (
        results_directory
        / "set_dqn_final_model.pt"
    )

    save_experiment_config(
        config_path,
        location_count,
        total_training_episodes,
        total_exposure_per_location,
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

                "order_size":
                    row[
                        "order_size"
                    ],

                "within_size_position":
                    row[
                        "within_size_position"
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

    probe_definition_rows = []

    for order_size in ORDER_SIZES:
        for row in probe_rows_by_size[
            order_size
        ]:
            probe_definition_rows.append(
                {
                    "order_size":
                        order_size,

                    "probe_index":
                        row[
                            "probe_index"
                        ],

                    "seed":
                        row[
                            "seed"
                        ],

                    "order":
                        str(
                            row[
                                "order"
                            ]
                        ),
                }
            )

    save_csv(
        probe_definition_rows,
        probe_definition_path,
    )

    # ========================================================
    # HISTORIES
    # ========================================================

    training_history = []

    aggregate_summaries = []

    per_size_summaries_history = []

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
    # EPISODE-0 EVALUATION
    # ========================================================

    print(
        "\nEvaluating transferred twenty-pick policy "
        "across 5/10/15/20 BEFORE mixed training..."
    )

    (
        initial_aggregate,
        initial_per_size,
        initial_raw,
    ) = evaluate_probe_sets(
        env,
        agent,
        probe_rows_by_size,
        all_pick_locations,
        episode=0,
        cycle=0,
    )

    aggregate_summaries.append(
        initial_aggregate
    )

    per_size_summaries_history.extend(
        initial_per_size
    )

    evaluation_raw_results.extend(
        initial_raw
    )

    best_evaluation_summary = (
        initial_aggregate.copy()
    )

    agent.save(
        best_model_path
    )

    print_mixed_evaluation(
        initial_aggregate,
        initial_per_size,
        "TRANSFERRED POLICY BEFORE MIXED TRAINING",
    )

    print(
        "\nEpisode 0 saved as initial best checkpoint candidate."
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

        order_size = schedule_row[
            "order_size"
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

        training_history.append(
            {
                "episode":
                    episode,

                "cycle":
                    cycle,

                "cycle_position":
                    cycle_position,

                "order_size":
                    order_size,

                "order":
                    str(
                        order
                    ),

                **metrics,

                "rolling_completion_rate":
                    rolling_completion_rate,

                "rolling_collection_fraction":
                    rolling_mean_collection,

                "rolling_invalid_action_rate":
                    rolling_mean_invalid,

                "rolling_reward":
                    rolling_mean_reward,
            }
        )

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
                f"Episode {episode:04d}/"
                f"{total_training_episodes}"
                f" | cycle {cycle:02d}/"
                f"{TRAIN_CYCLES}"
                f" | pos {cycle_position:03d}/"
                f"{ORDERS_PER_MIXED_CYCLE}"
                f" | size {order_size:02d}"
                f" | collect "
                f"{metrics['items_collected']}/"
                f"{order_size}"
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
        # MIXED DEVELOPMENT EVALUATION
        # ====================================================

        if episode % evaluate_every == 0:
            (
                aggregate_summary,
                per_size_summaries,
                evaluation_raw,
            ) = evaluate_probe_sets(
                env,
                agent,
                probe_rows_by_size,
                all_pick_locations,
                episode=episode,
                cycle=cycle,
            )

            aggregate_summaries.append(
                aggregate_summary
            )

            per_size_summaries_history.extend(
                per_size_summaries
            )

            evaluation_raw_results.extend(
                evaluation_raw
            )

            print_mixed_evaluation(
                aggregate_summary,
                per_size_summaries,
                (
                    "MIXED CONSOLIDATION EVALUATION "
                    f"AFTER CYCLE {cycle}/{TRAIN_CYCLES}"
                ),
            )

            if mixed_evaluation_is_better(
                aggregate_summary,
                best_evaluation_summary,
            ):
                best_evaluation_summary = (
                    aggregate_summary.copy()
                )

                agent.save(
                    best_model_path
                )

                print(
                    "\nNew best mixed-size consolidation model saved."
                )

            save_csv(
                training_history,
                training_history_path,
            )

            save_csv(
                aggregate_summaries,
                aggregate_summary_path,
            )

            save_csv(
                per_size_summaries_history,
                per_size_summary_path,
            )

            save_csv(
                evaluation_raw_results,
                evaluation_raw_path,
            )

    # ========================================================
    # FINISH
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
        aggregate_summaries,
        aggregate_summary_path,
    )

    save_csv(
        per_size_summaries_history,
        per_size_summary_path,
    )

    save_csv(
        evaluation_raw_results,
        evaluation_raw_path,
    )

    print(
        "\n"
        "================================================\n"
        "SET-DQN MIXED CONSOLIDATION COMPLETE\n"
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
        f"{total_exposure_per_location}"
    )

    print(
        f"Final epsilon: "
        f"{agent.epsilon:.4f}"
    )

    if rolling_completion:
        print(
            f"Final rolling completion: "
            f"{100.0 * float(np.mean(rolling_completion)):.1f}%"
        )

        print(
            f"Final rolling collection: "
            f"{float(np.mean(rolling_collection)):.3f}"
        )

        print(
            f"Final rolling invalid rate: "
            f"{float(np.mean(rolling_invalid_rate)):.1f}%"
        )

    print(
        "\nBest mixed-size checkpoint:"
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
        f"  Minimum completion: "
        f"{best_evaluation_summary['minimum_completion_rate']:.1f}%"
    )

    print(
        f"  Mean completion: "
        f"{best_evaluation_summary['mean_completion_rate']:.1f}%"
    )

    print(
        f"  Mean collection: "
        f"{best_evaluation_summary['mean_collection_fraction']:.3f}"
    )

    print(
        f"  Mean invalid rate: "
        f"{best_evaluation_summary['mean_invalid_action_rate']:.1f}%"
    )

    for order_size in ORDER_SIZES:
        print(
            f"  {order_size:2d}-pick completion: "
            f"{best_evaluation_summary[f'completion_{order_size}']:.1f}%"
            f" | collection "
            f"{best_evaluation_summary[f'collection_{order_size}']:.3f}"
            f" | invalid "
            f"{best_evaluation_summary[f'invalid_{order_size}']:.1f}%"
        )

    target_thresholds_met = all(
        [
            best_evaluation_summary[
                "completion_5"
            ] >= 85.0,

            best_evaluation_summary[
                "completion_10"
            ] >= 85.0,

            best_evaluation_summary[
                "completion_15"
            ] >= 80.0,

            best_evaluation_summary[
                "completion_20"
            ] >= 80.0,
        ]
    )

    print(
        "\nSuggested final-candidate threshold check:"
    )

    print(
        f"  5 >= 85, 10 >= 85, 15 >= 80, 20 >= 80: "
        f"{target_thresholds_met}"
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
        f"  Aggregate summary:\n"
        f"    {aggregate_summary_path}"
    )

    print(
        f"  Per-size summary:\n"
        f"    {per_size_summary_path}"
    )

    print(
        f"  Raw evaluation:\n"
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


if __name__ == "__main__":
    main()
