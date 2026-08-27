import random

import numpy as np
import pytest
import torch

from environment.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent


# ============================================================
# TEST WAREHOUSE
# ============================================================

def create_test_environment(max_steps=100):
    """
    Create the project's fixed 14 x 17 warehouse
    and return a WarehouseEnv instance.
    """

    height = 14
    width = 17

    grid = np.ones(
        (height, width),
        dtype=np.int8
    )

    # Front and rear cross aisles
    grid[0, :] = 0
    grid[13, :] = 0

    aisle_columns = [
        1, 3, 5, 7,
        9, 11, 13, 15
    ]

    for col in aisle_columns:
        grid[:, col] = 0

    depot = (0, 0)

    all_pick_locations = [
        (row, col)
        for col in aisle_columns
        for row in range(1, 13)
    ]

    env = WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=max_steps,
    )

    return env


def create_test_agent(
    env,
    batch_size=4,
):
    """
    Create a small DQN agent suitable for fast unit tests.

    These are test settings only, not final experimental
    hyperparameters.
    """

    agent = DQNAgent(
        env=env,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        batch_size=batch_size,
        buffer_capacity=100,
        target_update=10,
        hidden_dim=32,
    )

    return agent


def sample_order():
    """
    Fixed five-pick order used by several tests.
    """

    return [
        (2, 1),
        (8, 3),
        (5, 5),
        (10, 7),
        (4, 9),
    ]


# ============================================================
# STATE AND NETWORK SHAPE
# ============================================================

def test_environment_state_dimension():
    """
    State should contain:

        2 picker coordinates
        +
        96 binary remaining-pick indicators

    Total = 98
    """

    env = create_test_environment()

    state = env.reset(
        sample_order()
    )

    assert state.shape == (98,)
    assert state.dtype == np.float32


def test_policy_network_output_shape():
    """
    The network should map:

        98 state values
            ->
        4 Q-values

    corresponding to:
        0 = up
        1 = down
        2 = left
        3 = right
    """

    env = create_test_environment()
    agent = create_test_agent(env)

    state = env.reset(
        sample_order()
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = agent.policy_net(
            state_tensor
        )

    assert q_values.shape == (1, 4)


def test_target_network_output_shape():
    """
    Target network should have the same input/output
    structure as the policy network.
    """

    env = create_test_environment()
    agent = create_test_agent(env)

    state = env.reset(
        sample_order()
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = agent.target_net(
            state_tensor
        )

    assert q_values.shape == (1, 4)


# ============================================================
# INITIAL POLICY/TARGET SYNCHRONISATION
# ============================================================

def test_policy_and_target_network_initially_match():
    """
    The target network should initially contain the
    same parameters as the policy network.
    """

    env = create_test_environment()
    agent = create_test_agent(env)

    for policy_parameter, target_parameter in zip(
        agent.policy_net.parameters(),
        agent.target_net.parameters(),
    ):

        assert torch.allclose(
            policy_parameter,
            target_parameter,
        )


# ============================================================
# ACTION SELECTION
# ============================================================

def test_random_action_is_in_valid_range():
    """
    With epsilon = 1, actions are exploratory but must
    always remain in the valid action space 0..3.
    """

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = create_test_environment()
    agent = create_test_agent(env)

    agent.epsilon = 1.0

    state = env.reset(
        sample_order()
    )

    actions = [
        agent.select_action(state)
        for _ in range(100)
    ]

    for action in actions:

        assert isinstance(
            action,
            (int, np.integer),
        )

        assert 0 <= action < 4


def test_greedy_action_is_in_valid_range():
    """
    With epsilon = 0, the network should choose a greedy
    action and still return one of the four valid actions.
    """

    env = create_test_environment()
    agent = create_test_agent(env)

    agent.epsilon = 0.0

    state = env.reset(
        sample_order()
    )

    action = agent.select_action(
        state
    )

    assert isinstance(
        action,
        (int, np.integer),
    )

    assert 0 <= action < 4


def test_greedy_action_is_deterministic():
    """
    With epsilon disabled, the same state and unchanged
    network should produce the same action.
    """

    env = create_test_environment()
    agent = create_test_agent(env)

    agent.epsilon = 0.0

    state = env.reset(
        sample_order()
    )

    action_1 = agent.select_action(
        state
    )

    action_2 = agent.select_action(
        state
    )

    assert action_1 == action_2


# ============================================================
# REPLAY MEMORY + TRAINING
# ============================================================

def fill_replay_memory(
    env,
    agent,
    transitions=10,
):
    """
    Generate real transitions by allowing the agent to
    interact with WarehouseEnv.

    This tests the DQN/environment interface rather than
    inserting artificial transitions.
    """

    state = env.reset(
        sample_order()
    )

    for _ in range(transitions):

        action = random.randint(
            0,
            3
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(
            action
        )

        agent.remember(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )

        state = next_state

        if terminated or truncated:

            state = env.reset(
                sample_order()
            )


def test_train_step_waits_for_enough_experience():
    """
    Training should not occur before the replay buffer
    contains at least one complete batch.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    # Add fewer transitions than batch size.
    fill_replay_memory(
        env,
        agent,
        transitions=3,
    )

    loss = agent.train_step()

    assert loss is None


def test_train_step_returns_finite_loss():
    """
    Once sufficient transitions exist, train_step()
    should perform a gradient update and return a
    finite loss value.
    """

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    fill_replay_memory(
        env,
        agent,
        transitions=10,
    )

    loss = agent.train_step()

    assert loss is not None

    assert np.isfinite(
        loss
    )

    assert loss >= 0.0


def test_training_changes_policy_parameters():
    """
    A successful training step should modify at least
    one policy-network parameter.
    """

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    fill_replay_memory(
        env,
        agent,
        transitions=10,
    )

    parameters_before = [
        parameter.detach().clone()
        for parameter
        in agent.policy_net.parameters()
    ]

    loss = agent.train_step()

    assert loss is not None

    parameters_after = [
        parameter.detach().clone()
        for parameter
        in agent.policy_net.parameters()
    ]

    changed = any(
        not torch.equal(before, after)
        for before, after
        in zip(
            parameters_before,
            parameters_after,
        )
    )

    assert changed


# ============================================================
# EPSILON DECAY
# ============================================================

def test_epsilon_decays_after_training():
    """
    Successful training should reduce epsilon while
    respecting epsilon_min.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    fill_replay_memory(
        env,
        agent,
        transitions=10,
    )

    epsilon_before = agent.epsilon

    loss = agent.train_step()

    assert loss is not None

    assert agent.epsilon < epsilon_before

    assert agent.epsilon >= agent.epsilon_min


# ============================================================
# FULL ENVIRONMENT INTERACTION
# ============================================================

def test_agent_can_interact_with_environment():
    """
    The DQN should be capable of running an episode
    without shape/type/interface errors.

    The agent is untrained, so completion is NOT required.
    """

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = create_test_environment(
        max_steps=50
    )

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    state = env.reset(
        sample_order()
    )

    terminated = False
    truncated = False

    steps = 0

    while not (
        terminated
        or truncated
    ):

        action = agent.select_action(
            state
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

        agent.remember(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        )

        agent.train_step()

        state = next_state
        steps += 1

    assert steps > 0
    assert steps <= 50

    assert terminated or truncated

    assert state.shape == (98,)


# ============================================================
# MODEL SAVE / LOAD
# ============================================================

def test_model_save_and_load(tmp_path):
    """
    Saving and reloading a DQN should preserve the
    policy-network parameters.
    """

    env = create_test_environment()

    original_agent = create_test_agent(
        env
    )

    model_path = (
        tmp_path
        / "test_dqn_model.pt"
    )

    original_agent.save(
        model_path
    )

    loaded_agent = create_test_agent(
        env
    )

    loaded_agent.load(
        model_path
    )

    for original_parameter, loaded_parameter in zip(
        original_agent.policy_net.parameters(),
        loaded_agent.policy_net.parameters(),
    ):

        assert torch.allclose(
            original_parameter,
            loaded_parameter,
        )
