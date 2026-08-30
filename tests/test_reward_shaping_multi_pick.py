import numpy as np
import pytest

from environment.warehouse_env import WarehouseEnv

from utils.reward_shaping import (
    build_distance_lookup,
    calculate_potential_shaping_reward,
    calculate_routing_potential,
    decode_agent_position,
    get_remaining_pick_locations,
    minimum_spanning_tree_cost,
)


# ============================================================
# TEST WAREHOUSE
# ============================================================

def create_test_warehouse(
    max_steps=100,
):
    """
    Create the project's fixed 14 x 17 warehouse.
    """

    height = 14
    width = 17

    grid = np.ones(
        (height, width),
        dtype=np.int8,
    )

    # Front and rear cross aisles.
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

    env = WarehouseEnv(
        grid=grid,
        depot=depot,
        all_pick_locations=all_pick_locations,
        max_steps=max_steps,
    )

    distance_lookup = (
        build_distance_lookup(
            grid,
            depot,
            all_pick_locations,
        )
    )

    return (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    )


# ============================================================
# STATE DECODING
# ============================================================

def test_decode_agent_position_at_depot():
    """
    WarehouseEnv reset should decode to depot position.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    state = env.reset(
        [
            (8, 7)
        ]
    )

    position = decode_agent_position(
        state,
        grid.shape,
    )

    assert position == (
        0,
        0,
    )


def test_remaining_pick_decoder_handles_multiple_picks():
    """
    Multiple active order locations must be decoded correctly
    from the raw 98-dimensional environment state.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1),
        (4, 3),
        (8, 7),
    ]

    state = env.reset(
        order
    )

    remaining = (
        get_remaining_pick_locations(
            state,
            all_pick_locations,
        )
    )

    assert set(
        remaining
    ) == set(
        order
    )


# ============================================================
# MST
# ============================================================

def test_mst_cost_is_zero_for_one_pick():
    """
    A single remaining pick requires no connecting MST edge.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    cost = minimum_spanning_tree_cost(
        [
            (8, 7)
        ],
        distance_lookup,
    )

    assert cost == pytest.approx(
        0.0
    )


def test_mst_cost_for_three_known_locations():
    """
    Check a small hand-verifiable MST.

    Locations:
        A = (2,1)
        B = (2,3)
        C = (4,3)

    Warehouse shortest-path distances:

        A -> B = 6
        B -> C = 2
        A -> C = 8

    Therefore MST:

        6 + 2 = 8
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    points = [
        (2, 1),
        (2, 3),
        (4, 3),
    ]

    cost = minimum_spanning_tree_cost(
        points,
        distance_lookup,
    )

    assert cost == pytest.approx(
        8.0
    )


# ============================================================
# ONE-PICK BACKWARD COMPATIBILITY
# ============================================================

def test_one_pick_potential_matches_original_formula():
    """
    For one pick, MST cost is zero.

    Target:
        (8,7)

    Depot -> target:
        7 horizontal + 8 vertical = 15

    Target -> depot:
        15

    Therefore:

        Phi(s) = -(15 + 15) = -30

    This should reproduce the previous one-pick shaping
    formulation exactly.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    state = env.reset(
        [
            (8, 7)
        ]
    )

    potential = (
        calculate_routing_potential(
            state,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    assert potential == pytest.approx(
        -30.0
    )


# ============================================================
# TWO-PICK POTENTIAL
# ============================================================

def test_two_pick_potential_matches_hand_calculation():
    """
    Two-pick order:

        A = (2,1)
        B = (4,3)

    At depot:

        nearest entry distance = 3
        MST(A,B) = 8
        nearest depot exit = 3

    Estimated work:

        3 + 8 + 3 = 14

    Potential:

        -14
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1),
        (4, 3),
    ]

    state = env.reset(
        order
    )

    potential = (
        calculate_routing_potential(
            state,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    assert potential == pytest.approx(
        -14.0
    )


# ============================================================
# MOVEMENT SIGNAL
# ============================================================

def test_moving_towards_remaining_picks_improves_potential():
    """
    Moving right from depot toward (2,1) should reduce the
    estimated remaining work.

    Therefore the potential should become less negative.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1),
        (4, 3),
    ]

    state_before = env.reset(
        order
    )

    potential_before = (
        calculate_routing_potential(
            state_before,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    # Move right:
    # (0,0) -> (0,1)
    state_after, _, _, _, _ = env.step(
        3
    )

    potential_after = (
        calculate_routing_potential(
            state_after,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    assert potential_after > potential_before

    assert potential_before == pytest.approx(
        -14.0
    )

    assert potential_after == pytest.approx(
        -13.0
    )


def test_moving_towards_picks_has_positive_shaping_reward():
    """
    A move that improves routing potential should receive a
    positive potential-based shaping signal.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1),
        (4, 3),
    ]

    state_before = env.reset(
        order
    )

    state_after, _, _, _, _ = env.step(
        3
    )

    shaping_reward = (
        calculate_potential_shaping_reward(
            state_before,
            state_after,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
            gamma=0.99,
            scale=1.0,
        )
    )

    assert shaping_reward > 0.0

    # Phi before = -14
    # Phi after  = -13
    #
    # F = 0.99(-13) - (-14)
    #   = 1.13
    assert shaping_reward == pytest.approx(
        1.13
    )


def test_moving_away_from_nearest_pick_has_negative_shaping_reward():
    """
    From (0,1), moving right to (0,2) moves farther from the
    nearest pick (2,1).

    The shaping signal should therefore be negative.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1),
        (4, 3),
    ]

    state = env.reset(
        order
    )

    # Depot -> (0,1)
    state_close, _, _, _, _ = env.step(
        3
    )

    # (0,1) -> (0,2)
    state_farther, _, _, _, _ = env.step(
        3
    )

    shaping_reward = (
        calculate_potential_shaping_reward(
            state_close,
            state_farther,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
            gamma=0.99,
            scale=1.0,
        )
    )

    assert shaping_reward < 0.0


# ============================================================
# COLLECTION TRANSITION
# ============================================================

def test_potential_updates_after_collecting_one_of_two_picks():
    """
    After collecting (2,1), only (4,3) should remain.

    At position (2,1):

        distance to (4,3) = 8
        MST = 0
        (4,3) -> depot = 7

    Therefore:

        Phi = -(8 + 0 + 7)
            = -15
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1),
        (4, 3),
    ]

    state = env.reset(
        order
    )

    # (0,0) -> (0,1)
    state, _, _, _, _ = env.step(
        3
    )

    # -> (1,1)
    state, _, _, _, _ = env.step(
        1
    )

    # -> (2,1), collecting first pick
    state, _, terminated, truncated, _ = env.step(
        1
    )

    assert terminated is False
    assert truncated is False

    remaining = (
        get_remaining_pick_locations(
            state,
            all_pick_locations,
        )
    )

    assert remaining == [
        (4, 3)
    ]

    potential = (
        calculate_routing_potential(
            state,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    assert potential == pytest.approx(
        -15.0
    )


# ============================================================
# ALL PICKS COLLECTED
# ============================================================

def test_after_last_pick_potential_becomes_return_to_depot_distance():
    """
    Once the last pick has been collected, the potential
    should depend only on returning to the depot.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1)
    ]

    state = env.reset(
        order
    )

    # -> (0,1)
    state, _, _, _, _ = env.step(
        3
    )

    # -> (1,1)
    state, _, _, _, _ = env.step(
        1
    )

    # -> (2,1), collect
    state, _, terminated, truncated, _ = env.step(
        1
    )

    assert terminated is False
    assert truncated is False

    remaining = (
        get_remaining_pick_locations(
            state,
            all_pick_locations,
        )
    )

    assert remaining == []

    # Distance from (2,1) to depot:
    #
    # up 2 + left 1 = 3
    potential = (
        calculate_routing_potential(
            state,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    assert potential == pytest.approx(
        -3.0
    )


def test_successful_terminal_state_has_zero_potential():
    """
    At successful completion:

        all picks collected
        picker returned to depot

    therefore:

        Phi(s_terminal) = 0
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (1, 1)
    ]

    state = env.reset(
        order
    )

    # depot -> (0,1)
    state, _, _, _, _ = env.step(
        3
    )

    # -> (1,1), collect
    state, _, _, _, _ = env.step(
        1
    )

    # -> (0,1)
    state, _, _, _, _ = env.step(
        0
    )

    # -> depot, complete
    state, _, terminated, truncated, _ = env.step(
        2
    )

    assert terminated is True
    assert truncated is False

    potential = (
        calculate_routing_potential(
            state,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    assert potential == pytest.approx(
        0.0
    )


# ============================================================
# FINITE SHAPING
# ============================================================

def test_collection_transition_has_finite_shaping_reward():
    """
    Collecting one pick from a multi-pick order should produce
    a finite shaping reward.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    order = [
        (2, 1),
        (4, 3),
    ]

    state = env.reset(
        order
    )

    # Move to (1,1).
    state, _, _, _, _ = env.step(
        3
    )

    state_before_collection, _, _, _, _ = env.step(
        1
    )

    # Collect (2,1).
    state_after_collection, _, _, _, _ = env.step(
        1
    )

    shaping_reward = (
        calculate_potential_shaping_reward(
            state_before_collection,
            state_after_collection,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
            gamma=0.99,
            scale=1.0,
        )
    )

    assert np.isfinite(
        shaping_reward
    )


# ============================================================
# VALIDATION
# ============================================================

def test_incorrect_state_dimension_is_rejected():
    """
    Reward shaping should reject a raw state with the wrong
    number of location bits.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    incorrect_state = np.zeros(
        10,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        calculate_routing_potential(
            incorrect_state,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )


def test_invalid_gamma_is_rejected():
    """
    Potential-based shaping should reject an invalid discount
    factor.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
        distance_lookup,
    ) = create_test_warehouse()

    state = env.reset(
        [
            (8, 7)
        ]
    )

    with pytest.raises(
        ValueError
    ):

        calculate_potential_shaping_reward(
            state,
            state,
            grid.shape,
            depot,
            all_pick_locations,
            distance_lookup,
            gamma=1.5,
            scale=1.0,
        )
