import random

import numpy as np
import pytest
import torch

from agents.set_dqn_agent import (
    SetDQN,
    SetDQNAgent,
)

from environment.warehouse_env import WarehouseEnv


# ============================================================
# TEST ENVIRONMENT
# ============================================================

def create_test_environment(
    max_steps=150,
):
    """
    Create the project's fixed warehouse geometry.
    """

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

    return WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=max_steps,
        move_cost=-1.0,
        invalid_penalty=-2.0,
        pick_reward=2.0,
        completion_reward=20.0,
    )


def create_test_agent(
    env,
    batch_size=4,
    **kwargs,
):
    """
    Create CPU Set-DQN agent for deterministic unit tests.
    """

    return SetDQNAgent(
        env=env,
        batch_size=batch_size,
        buffer_capacity=500,
        target_update=10,
        device="cpu",
        **kwargs,
    )


# ============================================================
# NETWORK DIMENSIONS
# ============================================================

def test_raw_environment_state_dimension_is_98():
    """
    Set-DQN should consume the original WarehouseEnv state.
    """

    env = create_test_environment()

    state = env.reset(
        [
            (8, 7)
        ]
    )

    assert state.shape == (
        98,
    )


def test_set_network_output_shape_for_single_state():
    """
    One raw state should produce four movement Q-values.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = env.reset(
        [
            (8, 7)
        ]
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    )

    q_values = agent.policy_net(
        state_tensor
    )

    assert q_values.shape == (
        4,
    )


def test_set_network_output_shape_for_batch():
    """
    A state batch should produce [batch, 4] Q-values.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    states = []

    for order in [
        [(8, 7)],
        [(2, 1), (4, 3)],
        [(2, 1), (4, 3), (8, 7)],
    ]:

        states.append(
            env.reset(
                order
            )
        )

    state_batch = torch.tensor(
        np.stack(
            states
        ),
        dtype=torch.float32,
    )

    q_values = agent.policy_net(
        state_batch
    )

    assert q_values.shape == (
        3,
        4,
    )


# ============================================================
# SET INPUT CONSTRUCTION
# ============================================================

def test_build_set_inputs_has_expected_shapes():
    """
    The 96 warehouse locations should become 96 shared
    per-pick feature vectors.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = env.reset(
        [
            (2, 1),
            (4, 3),
            (8, 7),
        ]
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    )

    (
        picker,
        pick_features,
        active_mask,
    ) = agent.policy_net.build_set_inputs(
        state_tensor
    )

    assert picker.shape == (
        1,
        2,
    )

    assert pick_features.shape == (
        1,
        96,
        5,
    )

    assert active_mask.shape == (
        1,
        96,
    )


def test_active_mask_contains_one_entry_per_remaining_pick():
    """
    Three required locations should create three active set
    elements.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = env.reset(
        [
            (2, 1),
            (4, 3),
            (8, 7),
        ]
    )

    (
        _,
        _,
        active_mask,
    ) = agent.policy_net.build_set_inputs(
        state
    )

    assert float(
        active_mask.sum().item()
    ) == pytest.approx(
        3.0
    )


# ============================================================
# PERMUTATION INVARIANCE
# ============================================================

def test_set_network_is_permutation_invariant():
    """
    Reordering the pick-set dimension must not alter Q-values.

    This is the central property of the new architecture.
    """

    torch.manual_seed(
        123
    )

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = env.reset(
        [
            (2, 1),
            (4, 3),
            (8, 7),
            (10, 11),
            (12, 15),
        ]
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    )

    (
        picker,
        pick_features,
        active_mask,
    ) = agent.policy_net.build_set_inputs(
        state_tensor
    )

    q_original = (
        agent.policy_net.forward_from_set(
            picker,
            pick_features,
            active_mask,
        )
    )

    permutation = torch.randperm(
        pick_features.shape[
            1
        ]
    )

    permuted_features = (
        pick_features[
            :,
            permutation,
            :
        ]
    )

    permuted_mask = (
        active_mask[
            :,
            permutation
        ]
    )

    q_permuted = (
        agent.policy_net.forward_from_set(
            picker,
            permuted_features,
            permuted_mask,
        )
    )

    assert torch.allclose(
        q_original,
        q_permuted,
        atol=1e-6,
        rtol=1e-6,
    )


# ============================================================
# VARIABLE ORDER SIZES
# ============================================================

@pytest.mark.parametrize(
    "order_size",
    [
        1,
        2,
        3,
        5,
        10,
        20,
    ],
)
def test_same_network_accepts_variable_order_sizes(
    order_size,
):
    """
    One Set-DQN architecture must support all final project
    order sizes without introducing new input slots.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    order = (
        env.all_pick_locations[
            :order_size
        ]
    )

    state = env.reset(
        order
    )

    q_values = agent.get_q_values(
        state
    )

    assert q_values.shape == (
        4,
    )

    assert np.isfinite(
        q_values
    ).all()


def test_no_remaining_picks_produces_finite_q_values():
    """
    Once all picks are collected, mean pooling must remain
    numerically valid even though the active set is empty.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    # Construct a valid raw state representing:
    #
    # picker at depot
    # zero remaining picks
    state = np.zeros(
        98,
        dtype=np.float32,
    )

    q_values = agent.get_q_values(
        state
    )

    assert q_values.shape == (
        4,
    )

    assert np.isfinite(
        q_values
    ).all()


# ============================================================
# COLLECTION MASKING
# ============================================================

def test_collected_pick_is_removed_from_active_set():
    """
    After visiting a required location, that location must no
    longer contribute to the pooled remaining-pick set.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    order = [
        (1, 1),
        (4, 3),
    ]

    state = env.reset(
        order
    )

    (
        _,
        _,
        initial_mask,
    ) = agent.policy_net.build_set_inputs(
        state
    )

    assert initial_mask.sum().item() == pytest.approx(
        2.0
    )

    # Depot (0,0) -> (0,1)
    state, _, _, _, _ = env.step(
        3
    )

    # (0,1) -> (1,1), collect first pick
    state, _, terminated, truncated, _ = env.step(
        1
    )

    assert terminated is False
    assert truncated is False

    (
        _,
        _,
        new_mask,
    ) = agent.policy_net.build_set_inputs(
        state
    )

    assert new_mask.sum().item() == pytest.approx(
        1.0
    )

    location_index = (
        agent.policy_net
        .environment_state_locations
        .index(
            (1, 1)
        )
    )

    assert new_mask[
        0,
        location_index,
    ].item() == pytest.approx(
        0.0
    )


# ============================================================
# RELATIVE COORDINATES
# ============================================================

def test_relative_coordinates_update_with_picker_position():
    """
    Per-pick features must encode location relative to the
    current picker position.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    order = [
        (8, 7)
    ]

    state = env.reset(
        order
    )

    (
        _,
        features_before,
        _,
    ) = agent.policy_net.build_set_inputs(
        state
    )

    location_index = (
        agent.policy_net
        .environment_state_locations
        .index(
            (8, 7)
        )
    )

    relative_before = (
        features_before[
            0,
            location_index,
            2:4,
        ].clone()
    )

    # Move depot -> (0,1)
    state, _, _, _, _ = env.step(
        3
    )

    (
        _,
        features_after,
        _,
    ) = agent.policy_net.build_set_inputs(
        state
    )

    relative_after = (
        features_after[
            0,
            location_index,
            2:4,
        ]
    )

    assert not torch.allclose(
        relative_before,
        relative_after,
    )


# ============================================================
# POLICY / TARGET INITIALISATION
# ============================================================

def test_policy_and_target_network_initially_match():
    """
    Target network should begin as an exact policy copy.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    for (
        policy_parameter,
        target_parameter,
    ) in zip(
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
    Fully exploratory actions must be one of the four movement
    actions.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env,
        epsilon=1.0,
    )

    state = env.reset(
        [
            (8, 7)
        ]
    )

    for _ in range(
        100
    ):

        action = agent.select_action(
            state
        )

        assert action in {
            0,
            1,
            2,
            3,
        }


def test_greedy_action_is_deterministic():
    """
    Greedy evaluation should return the same action when the
    state and network parameters are unchanged.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = env.reset(
        [
            (2, 1),
            (8, 7),
            (12, 15),
        ]
    )

    actions = [
        agent.select_action(
            state,
            eval_mode=True,
        )
        for _ in range(
            20
        )
    ]

    assert len(
        set(
            actions
        )
    ) == 1


# ============================================================
# REPLAY / TRAINING
# ============================================================

def populate_replay(
    agent,
    env,
    count,
):
    """
    Populate replay with simple environment transitions.
    """

    random.seed(
        123
    )

    np.random.seed(
        123
    )

    state = env.reset(
        [
            (2, 1),
            (4, 3),
            (8, 7),
        ]
    )

    for _ in range(
        count
    ):

        action = random.randrange(
            4
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

        if (
            terminated
            or truncated
        ):

            state = env.reset(
                [
                    (2, 1),
                    (4, 3),
                    (8, 7),
                ]
            )

        else:

            state = next_state


def test_train_step_waits_for_enough_experience():
    """
    No optimisation should occur before batch_size transitions
    are available.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    populate_replay(
        agent,
        env,
        count=3,
    )

    loss = agent.train_step()

    assert loss is None


def test_train_step_returns_finite_loss():
    """
    A valid replay batch should produce a finite DQN loss.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    populate_replay(
        agent,
        env,
        count=10,
    )

    loss = agent.train_step()

    assert loss is not None

    assert np.isfinite(
        loss
    )


def test_training_changes_policy_parameters():
    """
    Gradient optimisation should modify at least one policy
    parameter.
    """

    torch.manual_seed(
        123
    )

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
    )

    populate_replay(
        agent,
        env,
        count=20,
    )

    before = [
        parameter
        .detach()
        .clone()
        for parameter
        in agent.policy_net.parameters()
    ]

    loss = agent.train_step()

    assert loss is not None

    after = [
        parameter
        .detach()
        .clone()
        for parameter
        in agent.policy_net.parameters()
    ]

    changed = any(
        not torch.allclose(
            before_parameter,
            after_parameter,
        )
        for (
            before_parameter,
            after_parameter,
        ) in zip(
            before,
            after,
        )
    )

    assert changed


def test_epsilon_decays_after_training():
    """
    Epsilon should decay after a successful gradient update.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=4,
        epsilon=1.0,
        epsilon_decay=0.9,
    )

    populate_replay(
        agent,
        env,
        count=10,
    )

    epsilon_before = (
        agent.epsilon
    )

    agent.train_step()

    assert agent.epsilon < epsilon_before

    assert agent.epsilon == pytest.approx(
        0.9
    )


# ============================================================
# TERMINATION / TRUNCATION SEMANTICS
# ============================================================

def create_dummy_state():
    """
    Create a valid empty raw WarehouseEnv state.
    """

    return np.zeros(
        98,
        dtype=np.float32,
    )


def test_successful_termination_is_stored_as_terminal():
    """
    Genuine success must mask Bellman bootstrapping.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = create_dummy_state()

    agent.remember(
        state,
        0,
        1.0,
        state,
        terminated=True,
        truncated=False,
    )

    stored_transition = (
        agent.memory.memory[
            -1
        ]
    )

    stored_terminated = (
        stored_transition[
            4
        ]
    )

    assert stored_terminated is True


def test_truncation_is_not_stored_as_terminal():
    """
    A max-step truncation ends rollout but must still permit
    Bellman bootstrapping.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = create_dummy_state()

    agent.remember(
        state,
        0,
        -1.0,
        state,
        terminated=False,
        truncated=True,
    )

    stored_transition = (
        agent.memory.memory[
            -1
        ]
    )

    stored_terminated = (
        stored_transition[
            4
        ]
    )

    assert stored_terminated is False


def test_normal_transition_is_not_terminal():
    """
    Ordinary transitions must also preserve bootstrapping.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = create_dummy_state()

    agent.remember(
        state,
        0,
        -1.0,
        state,
        terminated=False,
        truncated=False,
    )

    stored_terminated = (
        agent.memory.memory[
            -1
        ][
            4
        ]
    )

    assert stored_terminated is False


def test_replay_sample_preserves_terminal_mask():
    """
    Replay sampling must return:

        non-terminal -> 0
        terminal     -> 1
    """

    env = create_test_environment()

    agent = create_test_agent(
        env,
        batch_size=2,
    )

    state = create_dummy_state()

    agent.remember(
        state,
        0,
        0.0,
        state,
        terminated=False,
        truncated=True,
    )

    agent.remember(
        state,
        1,
        1.0,
        state,
        terminated=True,
        truncated=False,
    )

    (
        _,
        _,
        _,
        _,
        terminateds,
    ) = agent.memory.sample(
        2
    )

    assert sorted(
        terminateds.tolist()
    ) == [
        0.0,
        1.0,
    ]


# ============================================================
# TARGET UPDATE
# ============================================================

def test_target_network_updates_when_interval_reached():
    """
    With target_update=1, the target network should synchronise
    after every gradient update.
    """

    env = create_test_environment()

    agent = SetDQNAgent(
        env=env,
        batch_size=4,
        buffer_capacity=500,
        target_update=1,
        device="cpu",
    )

    populate_replay(
        agent,
        env,
        count=10,
    )

    agent.train_step()

    for (
        policy_parameter,
        target_parameter,
    ) in zip(
        agent.policy_net.parameters(),
        agent.target_net.parameters(),
    ):

        assert torch.allclose(
            policy_parameter,
            target_parameter,
        )


# ============================================================
# SAVE / LOAD
# ============================================================

def test_model_save_and_load(
    tmp_path,
):
    """
    Saving and reloading should reproduce identical Q-values.
    """

    torch.manual_seed(
        123
    )

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    state = env.reset(
        [
            (2, 1),
            (4, 3),
            (8, 7),
        ]
    )

    # Deliberately modify the policy so this is not merely
    # checking default initialisation.
    with torch.no_grad():

        for parameter in agent.policy_net.parameters():

            parameter.add_(
                0.01
            )

    q_before = agent.get_q_values(
        state
    )

    checkpoint = (
        tmp_path
        / "set_dqn_test.pt"
    )

    agent.save(
        checkpoint
    )

    new_agent = create_test_agent(
        env
    )

    new_agent.load(
        checkpoint
    )

    q_after = new_agent.get_q_values(
        state
    )

    np.testing.assert_allclose(
        q_before,
        q_after,
        rtol=1e-6,
        atol=1e-6,
    )


# ============================================================
# VALIDATION
# ============================================================

def test_network_rejects_incorrect_raw_state_dimension():
    """
    Set-DQN must reject states that do not match WarehouseEnv's
    98-dimensional state.
    """

    env = create_test_environment()

    agent = create_test_agent(
        env
    )

    incorrect_state = np.zeros(
        62,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        agent.policy_net(
            incorrect_state
        )


def test_network_rejects_invalid_max_order_size():
    """
    max_order_size must be positive.
    """

    env = create_test_environment()

    with pytest.raises(
        ValueError
    ):

        SetDQN(
            all_pick_locations=env.all_pick_locations,
            grid_shape=env.grid.shape,
            max_order_size=0,
        )


def test_network_rejects_duplicate_location_definitions():
    """
    Duplicate warehouse locations would make state decoding
    ambiguous.
    """

    env = create_test_environment()

    locations = list(
        env.all_pick_locations
    )

    locations.append(
        locations[
            0
        ]
    )

    with pytest.raises(
        ValueError
    ):

        SetDQN(
            all_pick_locations=locations,
            grid_shape=env.grid.shape,
        )
