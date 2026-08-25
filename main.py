import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent
from utils.order_generation import generate_order

from heuristics.s_shape import s_shape_route
from heuristics.return_route import return_route
from heuristics.largest_gap import largest_gap_route

# Use this once your final Ratliff-Rosenthal solver is implemented.
# from exact.ratliff_rosenthal import exact_optimal_distance


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

ORDER_SIZES = [5, 10, 15, 20]

DISTRIBUTIONS = [
    "uniform",
    "clustered"
]

TRAIN_EPISODES = 5000

TEST_ORDERS_PER_CONDITION = 50

MAX_STEPS = 500

TRAIN_SEED = 42
TEST_SEED = 1000


# ============================================================
# WAREHOUSE CREATION
# ============================================================

def create_warehouse():
    """
    Create the final fixed single-block rectangular warehouse.

    Grid:
        14 rows x 17 columns

    Front cross aisle:
        row 0

    Rear cross aisle:
        row 13

    Picking aisles:
        columns 1, 3, 5, 7, 9, 11, 13, 15

    Pick positions:
        rows 1-12

    Depot:
        (0, 0)

    Grid values:
        0 = walkable
        1 = obstacle
    """

    height = 14
    width = 17

    grid = np.ones(
        (height, width),
        dtype=np.int8
    )

    # --------------------------------------------------------
    # Front cross aisle
    # --------------------------------------------------------

    grid[0, :] = 0

    # --------------------------------------------------------
    # Rear cross aisle
    # --------------------------------------------------------

    grid[13, :] = 0

    # --------------------------------------------------------
    # Eight vertical picking aisles
    # --------------------------------------------------------

    aisle_columns = [
        1, 3, 5, 7,
        9, 11, 13, 15
    ]

    for col in aisle_columns:
        grid[:, col] = 0

    # --------------------------------------------------------
    # Depot
    # --------------------------------------------------------

    depot = (0, 0)

    # --------------------------------------------------------
    # 96 fixed pick locations
    # --------------------------------------------------------

    all_pick_locations = []

    for col in aisle_columns:
        for row in range(1, 13):
            all_pick_locations.append(
                (row, col)
            )

    return (
        grid,
        depot,
        all_pick_locations,
        aisle_columns
    )


# ============================================================
# RANDOM SEEDS
# ============================================================

def set_random_seeds(seed):
    """
    Set Python, NumPy and PyTorch seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DQN TRAINING
# ============================================================

def train_agent(
    env,
    agent,
    all_pick_locations,
    aisle_columns,
    n_episodes=TRAIN_EPISODES,
    seed=TRAIN_SEED
):
    """
    Train one DQN on uniformly generated orders.

    Training episodes contain a mixture of:
        5, 10, 15 and 20 pick orders.

    The DQN therefore learns one policy that can later
    be evaluated separately across all four order sizes.
    """

    rng = np.random.default_rng(seed)

    episode_rewards = []
    episode_distances = []
    completion_history = []
    losses = []

    for episode in range(n_episodes):

        # ----------------------------------------------------
        # Randomly choose order size
        # ----------------------------------------------------

        order_size = int(
            rng.choice(ORDER_SIZES)
        )

        # Generate a unique seed for this training order.
        order_seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        # ----------------------------------------------------
        # Training orders are uniform only
        # ----------------------------------------------------

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=order_size,
            distribution="uniform",
            seed=order_seed,
            aisle_columns=aisle_columns
        )

        state = env.reset(order)

        episode_reward = 0.0

        terminated = False
        truncated = False

        # ----------------------------------------------------
        # Run one episode
        # ----------------------------------------------------

        while not (
            terminated or truncated
        ):

            action = agent.select_action(
                state
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            agent.remember(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated
            )

            loss = agent.train_step()

            if loss is not None:
                losses.append(loss)

            episode_reward += reward

            state = next_state

        # ----------------------------------------------------
        # Record training information
        # ----------------------------------------------------

        episode_rewards.append(
            episode_reward
        )

        episode_distances.append(
            env.travel_distance
        )

        completion_history.append(
            int(terminated)
        )

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        if (episode + 1) % 100 == 0:

            recent_completion = np.mean(
                completion_history[-100:]
            )

            recent_reward = np.mean(
                episode_rewards[-100:]
            )

            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Completion: {recent_completion:.2%} | "
                f"Mean reward: {recent_reward:.2f}"
            )

    training_history = {
        "reward": episode_rewards,
        "distance": episode_distances,
        "completed": completion_history,
        "loss": losses
    }

    return training_history


# ============================================================
# DQN EVALUATION
# ============================================================

def evaluate_dqn(
    env,
    agent,
    order
):
    """
    Evaluate the trained DQN greedily.

    Exploration is disabled during evaluation.
    """

    state = env.reset(order)

    terminated = False
    truncated = False

    start_time = time.perf_counter()

    while not (
        terminated or truncated
    ):

        action = agent.select_action(
            state,
            eval_mode=True
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        state = next_state

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "distance": env.travel_distance,
        "completed": terminated,
        "truncated": truncated,
        "runtime": runtime,
        "steps": env.steps
    }


# ============================================================
# HEURISTIC EVALUATION
# ============================================================

def evaluate_heuristic(
    heuristic_function,
    grid,
    depot,
    picks
):
    """
    Run one deterministic heuristic and record runtime.

    The heuristic function is expected to return
    total route distance.
    """

    start_time = time.perf_counter()

    distance = heuristic_function(
        grid,
        depot,
        picks
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "distance": distance,
        "completed": True,
        "truncated": False,
        "runtime": runtime,
        "steps": np.nan
    }


# ============================================================
# EXACT SOLVER EVALUATION
# ============================================================

def evaluate_exact(
    grid,
    depot,
    picks
):
    """
    Determine the optimal distance for one order.
    """

    start_time = time.perf_counter()

    distance = exact_optimal_distance(
        grid,
        depot,
        picks
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    return {
        "distance": distance,
        "completed": True,
        "truncated": False,
        "runtime": runtime,
        "steps": np.nan
    }


# ============================================================
# TEST ORDER GENERATION
# ============================================================

def generate_test_orders(
    all_pick_locations,
    aisle_columns,
    order_size,
    distribution,
    number_of_orders,
    seed
):
    """
    Generate a fixed test set for one experimental condition.

    These exact orders will later be used by every method.
    """

    rng = np.random.default_rng(seed)

    orders = []

    seen_orders = set()

    while len(orders) < number_of_orders:

        order_seed = int(
            rng.integers(
                0,
                2**32 - 1
            )
        )

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=order_size,
            distribution=distribution,
            seed=order_seed,
            aisle_columns=aisle_columns
        )

        # Sort only for duplicate detection.
        order_key = tuple(
            sorted(order)
        )

        if order_key not in seen_orders:

            seen_orders.add(
                order_key
            )

            orders.append(
                order
            )

    return orders


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_random_seeds(
        TRAIN_SEED
    )

    # --------------------------------------------------------
    # Warehouse
    # --------------------------------------------------------

    (
        grid,
        depot,
        all_pick_locations,
        aisle_columns
    ) = create_warehouse()

    print(
        f"Warehouse grid: {grid.shape}"
    )

    print(
        f"Number of pick locations: "
        f"{len(all_pick_locations)}"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=MAX_STEPS
    )

    # --------------------------------------------------------
    # DQN agent
    # --------------------------------------------------------

    agent = DQNAgent(
        env=env
    )

    # --------------------------------------------------------
    # TRAIN ONE DQN
    # --------------------------------------------------------

    print("\nStarting DQN training...\n")

    training_start = time.perf_counter()

    training_history = train_agent(
        env=env,
        agent=agent,
        all_pick_locations=all_pick_locations,
        aisle_columns=aisle_columns,
        n_episodes=TRAIN_EPISODES,
        seed=TRAIN_SEED
    )

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    print(
        f"\nTraining finished in "
        f"{total_training_time:.2f} seconds."
    )

    # --------------------------------------------------------
    # Freeze policy for evaluation
    # --------------------------------------------------------

    agent.policy_net.eval()

    # No exploration during final evaluation.
    agent.epsilon = 0.0

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    results_directory = Path(
        "results"
    )

    results_directory.mkdir(
        exist_ok=True
    )

    agent.save(
        results_directory
        / "dqn_policy.pt"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    raw_results = []

    condition_number = 0

    for order_size in ORDER_SIZES:

        for distribution in DISTRIBUTIONS:

            condition_number += 1

            print(
                f"\nEvaluating: "
                f"size={order_size}, "
                f"distribution={distribution}"
            )

            # Different deterministic seed
            # for each experimental condition.
            condition_seed = (
                TEST_SEED
                + condition_number
            )

            test_orders = generate_test_orders(
                all_pick_locations=all_pick_locations,
                aisle_columns=aisle_columns,
                order_size=order_size,
                distribution=distribution,
                number_of_orders=TEST_ORDERS_PER_CONDITION,
                seed=condition_seed
            )

            # ------------------------------------------------
            # Same orders used by EVERY method
            # ------------------------------------------------

            for order_index, picks in enumerate(
                test_orders
            ):

                order_id = (
                    f"{distribution}_"
                    f"{order_size}_"
                    f"{order_index:03d}"
                )

                # ============================================
                # EXACT OPTIMUM
                # ============================================

                optimal_result = evaluate_exact(
                    grid,
                    depot,
                    picks
                )

                optimal_distance = (
                    optimal_result[
                        "distance"
                    ]
                )

                # Store exact result.
                raw_results.append(
                    {
                        "order_id": order_id,
                        "order_size": order_size,
                        "distribution": distribution,
                        "method": "exact",
                        "distance": optimal_distance,
                        "optimality_gap": 0.0,
                        "completed": True,
                        "truncated": False,
                        "runtime": optimal_result[
                            "runtime"
                        ],
                        "steps": np.nan
                    }
                )

                # ============================================
                # DQN
                # ============================================

                dqn_result = evaluate_dqn(
                    env,
                    agent,
                    picks
                )

                if dqn_result["completed"]:

                    dqn_gap = (
                        (
                            dqn_result["distance"]
                            - optimal_distance
                        )
                        / optimal_distance
                    ) * 100

                else:
                    dqn_gap = np.nan

                raw_results.append(
                    {
                        "order_id": order_id,
                        "order_size": order_size,
                        "distribution": distribution,
                        "method": "dqn",
                        "distance": dqn_result[
                            "distance"
                        ],
                        "optimality_gap": dqn_gap,
                        "completed": dqn_result[
                            "completed"
                        ],
                        "truncated": dqn_result[
                            "truncated"
                        ],
                        "runtime": dqn_result[
                            "runtime"
                        ],
                        "steps": dqn_result[
                            "steps"
                        ]
                    }
                )

                # ============================================
                # HEURISTICS
                # ============================================

                heuristic_methods = {
                    "s_shape": s_shape_route,
                    "return": return_route,
                    "largest_gap": largest_gap_route
                }

                for (
                    method_name,
                    method_function
                ) in heuristic_methods.items():

                    result = evaluate_heuristic(
                        method_function,
                        grid,
                        depot,
                        picks
                    )

                    gap = (
                        (
                            result["distance"]
                            - optimal_distance
                        )
                        / optimal_distance
                    ) * 100

                    raw_results.append(
                        {
                            "order_id": order_id,
                            "order_size": order_size,
                            "distribution": distribution,
                            "method": method_name,
                            "distance": result[
                                "distance"
                            ],
                            "optimality_gap": gap,
                            "completed": True,
                            "truncated": False,
                            "runtime": result[
                                "runtime"
                            ],
                            "steps": np.nan
                        }
                    )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        raw_results
    )

    raw_results_path = (
        results_directory
        / "raw_results.csv"
    )

    results_df.to_csv(
        raw_results_path,
        index=False
    )

    # ========================================================
    # SUMMARY RESULTS
    # ========================================================

    summary = (
        results_df
        .groupby(
            [
                "order_size",
                "distribution",
                "method"
            ]
        )
        .agg(
            mean_distance=(
                "distance",
                "mean"
            ),
            std_distance=(
                "distance",
                "std"
            ),
            mean_optimality_gap=(
                "optimality_gap",
                "mean"
            ),
            mean_runtime=(
                "runtime",
                "mean"
            ),
            completion_rate=(
                "completed",
                "mean"
            )
        )
        .reset_index()
    )

    summary_path = (
        results_directory
        / "summary_results.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # TRAINING INFORMATION
    # ========================================================

    training_df = pd.DataFrame({
        "episode": np.arange(
            1,
            len(
                training_history[
                    "reward"
                ]
            ) + 1
        ),
        "reward": training_history[
            "reward"
        ],
        "distance": training_history[
            "distance"
        ],
        "completed": training_history[
            "completed"
        ]
    })

    training_df.to_csv(
        results_directory
        / "training_history.csv",
        index=False
    )

    print(
        "\nExperiment complete."
    )

    print(
        f"Raw results saved to: "
        f"{raw_results_path}"
    )

    print(
        f"Summary results saved to: "
        f"{summary_path}"
    )

    return (
        results_df,
        summary,
        training_history
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_experiment()
