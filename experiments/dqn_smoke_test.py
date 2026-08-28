"""
DQN smoke-training test.

Purpose
-------
Check that the complete reinforcement-learning pipeline works
end-to-end before running longer development or final training.

This script is NOT a final experiment.

It checks that:

    order generation
        ->
    environment reset
        ->
    DQN action selection
        ->
    environment step
        ->
    replay buffer
        ->
    gradient update
        ->
    episode termination/truncation
        ->
    model saving

all work together without errors.
"""

from utils.order_generation import generate_order
from agents.dqn_agent import DQNAgent
from environment.warehouse_env import WarehouseEnv
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# PROJECT IMPORT PATH
# ============================================================
#
# This allows the script to be run directly using:
#
#     python experiments/dqn_smoke_test.py
#
# while still importing modules from the project root.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SMOKE-TEST SETTINGS
# ============================================================

SEED = 42

SMOKE_EPISODES = 50

ORDER_SIZE = 5

MAX_STEPS = 200

DISTRIBUTION = "uniform"


# ============================================================
# WAREHOUSE
# ============================================================

def create_warehouse():
    """
    Create the fixed warehouse used by the project.

    Warehouse:
        - 14 rows
        - 17 columns
        - front cross aisle at row 0
        - rear cross aisle at row 13
        - 8 picking aisles
        - 12 pick positions per aisle
        - 96 fixed pick locations
        - depot at (0, 0)

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

    # Front and rear cross aisles
    grid[0, :] = 0
    grid[13, :] = 0

    aisle_columns = [
        1, 3, 5, 7,
        9, 11, 13, 15,
    ]

    # Vertical picking aisles
    for col in aisle_columns:
        grid[:, col] = 0

    depot = (0, 0)

    # 8 aisles x 12 storage positions = 96 locations
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
# RANDOM SEEDS
# ============================================================

def set_random_seeds(seed):
    """
    Make the smoke run reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment(
    grid,
    depot,
    all_pick_locations,
):
    """
    Create the WarehouseEnv used for smoke training.
    """

    return WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=MAX_STEPS,

        # Provisional development reward values.
        move_cost=-1.0,
        invalid_penalty=-2.0,
        pick_reward=2.0,
        completion_reward=20.0,
    )


# ============================================================
# CREATE AGENT
# ============================================================

def create_agent(env):
    """
    Create the DQN agent.

    These parameters are development settings only.
    They are not being treated as final experimental
    hyperparameters.
    """

    return DQNAgent(
        env=env,

        lr=1e-4,
        gamma=0.99,

        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,

        batch_size=64,
        buffer_capacity=10000,

        target_update=1000,

        hidden_dim=256,
    )


# ============================================================
# RUN ONE EPISODE
# ============================================================

def run_training_episode(
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

    state = env.reset(
        order
    )

    terminated = False
    truncated = False

    total_reward = 0.0

    losses = []

    while not (
        terminated
        or truncated
    ):

        # ----------------------------------------------------
        # Choose action
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
        # Store experience
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
        # DQN update
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

        # ----------------------------------------------------
        # Move to next state
        # ----------------------------------------------------

        state = next_state

        total_reward += reward

    # --------------------------------------------------------
    # Episode metrics
    # --------------------------------------------------------

    completed = bool(
        terminated
    )

    average_loss = (
        float(np.mean(losses))
        if losses
        else np.nan
    )

    return {
        "reward": float(total_reward),
        "distance": float(env.travel_distance),
        "steps": int(env.steps),
        "completed": completed,
        "truncated": bool(truncated),
        "epsilon": float(agent.epsilon),
        "average_loss": average_loss,
        "training_updates": len(losses),
    }


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(
    history,
    output_path,
):
    """
    Save episode-level smoke-test metrics to CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "episode",
        "reward",
        "distance",
        "steps",
        "completed",
        "truncated",
        "epsilon",
        "average_loss",
        "training_updates",
    ]

    with output_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# MAIN SMOKE TEST
# ============================================================

def main():

    print(
        "\n"
        "========================================\n"
        "DQN SMOKE TRAINING TEST\n"
        "========================================"
    )

    print(
        f"Episodes:      {SMOKE_EPISODES}\n"
        f"Order size:    {ORDER_SIZE}\n"
        f"Distribution:  {DISTRIBUTION}\n"
        f"Max steps:     {MAX_STEPS}\n"
        f"Seed:          {SEED}\n"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_random_seeds(
        SEED
    )

    # --------------------------------------------------------
    # Warehouse
    # --------------------------------------------------------

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    ) = create_warehouse()

    print(
        f"Warehouse grid:      {grid.shape}"
    )

    print(
        f"Pick locations:      "
        f"{len(all_pick_locations)}"
    )

    # --------------------------------------------------------
    # Environment + agent
    # --------------------------------------------------------

    env = create_environment(
        grid,
        depot,
        all_pick_locations,
    )

    agent = create_agent(
        env
    )

    print(
        f"DQN state dimension: "
        f"{2 + len(all_pick_locations)}"
    )

    print(
        "DQN actions:         4"
    )

    print(
        "\nStarting smoke training...\n"
    )

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(
        1,
        SMOKE_EPISODES + 1,
    ):

        # ----------------------------------------------------
        # Generate deterministic uniform order
        # ----------------------------------------------------

        order = generate_order(
            all_pick_locations=all_pick_locations,
            size=ORDER_SIZE,
            distribution=DISTRIBUTION,
            seed=SEED + episode,
            aisle_columns=aisle_columns,
        )

        # ----------------------------------------------------
        # Run episode
        # ----------------------------------------------------

        metrics = run_training_episode(
            env,
            agent,
            order,
        )

        metrics["episode"] = episode

        history.append(
            metrics
        )

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        loss_text = (
            f"{metrics['average_loss']:.4f}"
            if np.isfinite(
                metrics["average_loss"]
            )
            else "N/A"
        )

        status = (
            "COMPLETED"
            if metrics["completed"]
            else "TRUNCATED"
        )

        print(
            f"Episode "
            f"{episode:02d}/{SMOKE_EPISODES}"
            f" | {status:<9}"
            f" | reward "
            f"{metrics['reward']:8.1f}"
            f" | distance "
            f"{metrics['distance']:6.0f}"
            f" | steps "
            f"{metrics['steps']:3d}"
            f" | eps "
            f"{metrics['epsilon']:.3f}"
            f" | loss "
            f"{loss_text}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    completed_count = sum(
        episode["completed"]
        for episode in history
    )

    completion_rate = (
        100.0
        * completed_count
        / len(history)
    )

    average_reward = np.mean(
        [
            episode["reward"]
            for episode in history
        ]
    )

    average_distance = np.mean(
        [
            episode["distance"]
            for episode in history
        ]
    )

    average_steps = np.mean(
        [
            episode["steps"]
            for episode in history
        ]
    )

    total_updates = sum(
        episode["training_updates"]
        for episode in history
    )

    print(
        "\n"
        "========================================\n"
        "SMOKE TEST SUMMARY\n"
        "========================================"
    )

    print(
        f"Completed episodes: "
        f"{completed_count}/{SMOKE_EPISODES}"
    )

    print(
        f"Completion rate:    "
        f"{completion_rate:.1f}%"
    )

    print(
        f"Average reward:     "
        f"{average_reward:.2f}"
    )

    print(
        f"Average distance:   "
        f"{average_distance:.2f}"
    )

    print(
        f"Average steps:      "
        f"{average_steps:.2f}"
    )

    print(
        f"Training updates:   "
        f"{total_updates}"
    )

    print(
        f"Final epsilon:      "
        f"{agent.epsilon:.4f}"
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    results_directory = (
        PROJECT_ROOT
        / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        results_directory
        / "dqn_smoke_history.csv"
    )

    model_path = (
        results_directory
        / "dqn_smoke_model.pt"
    )

    save_history(
        history,
        history_path,
    )

    agent.save(
        model_path
    )

    print(
        "\nSaved:"
    )

    print(
        f"  History: {history_path}"
    )

    print(
        f"  Model:   {model_path}"
    )

    print(
        "\nSmoke training completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
