import numpy as np
import pytest

from environment.warehouse_env import WarehouseEnv
from utils.state_encoding import CoordinateStateEncoder


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

    # Front and rear cross aisles
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

    return (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    )


def create_encoder(
    all_pick_locations,
    grid,
    max_order_size=20,
):
    """
    Create the coordinate encoder used by the tests.
    """

    return CoordinateStateEncoder(
        all_pick_locations=all_pick_locations,
        grid_shape=grid.shape,
        max_order_size=max_order_size,
    )


# ============================================================
# OUTPUT DIMENSION
# ============================================================

def test_coordinate_state_dimension_is_62():
    """
    Final representation should contain:

        2 picker coordinates
        +
        20 x 3 pick-slot values

        = 62
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    assert encoder.output_dim == 62

    assert encoder.get_output_dim() == 62


# ============================================================
# OUTPUT TYPE
# ============================================================

def test_encoded_state_is_float32():
    """
    Encoded state should use float32 for PyTorch input.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    order = [
        (8, 7)
    ]

    environment_state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    encoded_state = encoder.encode(
        environment_state
    )

    assert encoded_state.dtype == np.float32


# ============================================================
# PICKER POSITION
# ============================================================

def test_picker_coordinates_are_preserved():
    """
    The first two encoded values should remain exactly the
    normalised picker coordinates supplied by WarehouseEnv.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    order = [
        (8, 7)
    ]

    environment_state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    encoded_state = encoder.encode(
        environment_state
    )

    assert encoded_state[0] == pytest.approx(
        environment_state[0]
    )

    assert encoded_state[1] == pytest.approx(
        environment_state[1]
    )


# ============================================================
# ONE-PICK ENCODING
# ============================================================

def test_one_pick_is_encoded_as_coordinates_and_active_flag():
    """
    A one-pick order at (8, 7) should appear in the first
    pick slot as:

        8 / 13
        7 / 16
        1
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    target = (
        8,
        7,
    )

    order = [
        target
    ]

    environment_state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    encoded_state = encoder.encode(
        environment_state
    )

    # First pick slot starts at index 2.
    assert encoded_state[2] == pytest.approx(
        8.0 / 13.0
    )

    assert encoded_state[3] == pytest.approx(
        7.0 / 16.0
    )

    assert encoded_state[4] == pytest.approx(
        1.0
    )


# ============================================================
# UNUSED SLOTS
# ============================================================

def test_unused_pick_slots_are_zero():
    """
    For a one-pick order, slots 2..20 should remain padded
    with zeros.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    order = [
        (8, 7)
    ]

    environment_state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    encoded_state = encoder.encode(
        environment_state
    )

    # First slot occupies indices 2,3,4.
    # Everything afterwards should be zero.
    assert np.allclose(
        encoded_state[5:],
        0.0,
    )


# ============================================================
# COLLECTED PICK
# ============================================================

def test_collected_pick_slot_becomes_zero():
    """
    Once a pick is collected its assigned slot should become:

        [0, 0, 0]
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    target = (
        2,
        1,
    )

    order = [
        target
    ]

    state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    # Initial target is active.
    initial_encoded = encoder.encode(
        state
    )

    assert initial_encoded[4] == pytest.approx(
        1.0
    )

    # --------------------------------------------------------
    # Route:
    #
    # depot (0,0)
    # right -> (0,1)
    # down  -> (1,1)
    # down  -> (2,1) and collect target
    #
    # Actions:
    # 1 = down
    # 3 = right
    # --------------------------------------------------------

    state, _, _, _, _ = env.step(
        3
    )

    state, _, _, _, _ = env.step(
        1
    )

    state, _, _, _, _ = env.step(
        1
    )

    encoded_after_collection = (
        encoder.encode(
            state
        )
    )

    assert np.allclose(
        encoded_after_collection[
            2:5
        ],
        [
            0.0,
            0.0,
            0.0,
        ],
    )


# ============================================================
# SLOT STABILITY
# ============================================================

def test_other_pick_does_not_shift_after_collection():
    """
    Pick slots must remain stable throughout an episode.

    For the order:

        (2,1)
        (4,3)

    collecting (2,1) should clear slot 1 but leave (4,3)
    in slot 2 rather than shifting it into slot 1.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    order = [
        (2, 1),
        (4, 3),
    ]

    state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    initial_encoded = encoder.encode(
        state
    )

    # --------------------------------------------------------
    # Slot 1 = (2,1)
    # --------------------------------------------------------

    assert initial_encoded[2] == pytest.approx(
        2.0 / 13.0
    )

    assert initial_encoded[3] == pytest.approx(
        1.0 / 16.0
    )

    assert initial_encoded[4] == pytest.approx(
        1.0
    )

    # --------------------------------------------------------
    # Slot 2 = (4,3)
    #
    # Slot 2 starts at index:
    #   2 + (3 * 1) = 5
    # --------------------------------------------------------

    assert initial_encoded[5] == pytest.approx(
        4.0 / 13.0
    )

    assert initial_encoded[6] == pytest.approx(
        3.0 / 16.0
    )

    assert initial_encoded[7] == pytest.approx(
        1.0
    )

    # --------------------------------------------------------
    # Collect first pick at (2,1)
    # --------------------------------------------------------

    state, _, _, _, _ = env.step(
        3
    )

    state, _, _, _, _ = env.step(
        1
    )

    state, _, _, _, _ = env.step(
        1
    )

    encoded_after_collection = (
        encoder.encode(
            state
        )
    )

    # Slot 1 cleared.
    assert np.allclose(
        encoded_after_collection[
            2:5
        ],
        [
            0.0,
            0.0,
            0.0,
        ],
    )

    # Slot 2 remains in the SAME slot.
    assert encoded_after_collection[
        5
    ] == pytest.approx(
        4.0 / 13.0
    )

    assert encoded_after_collection[
        6
    ] == pytest.approx(
        3.0 / 16.0
    )

    assert encoded_after_collection[
        7
    ] == pytest.approx(
        1.0
    )


# ============================================================
# DETERMINISTIC SLOT ORDER
# ============================================================

def test_order_input_order_does_not_change_encoding():
    """
    Supplying the same set of picks in a different Python
    list order should produce the same slot assignments.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder_1 = create_encoder(
        all_pick_locations,
        grid,
    )

    encoder_2 = create_encoder(
        all_pick_locations,
        grid,
    )

    order_1 = [
        (8, 7),
        (2, 1),
        (4, 3),
    ]

    order_2 = [
        (4, 3),
        (8, 7),
        (2, 1),
    ]

    state_1 = env.reset(
        order_1
    )

    encoder_1.reset(
        order_1
    )

    encoded_1 = encoder_1.encode(
        state_1
    )

    state_2 = env.reset(
        order_2
    )

    encoder_2.reset(
        order_2
    )

    encoded_2 = encoder_2.encode(
        state_2
    )

    assert np.allclose(
        encoded_1,
        encoded_2,
    )


# ============================================================
# MAXIMUM ORDER SIZE
# ============================================================

def test_twenty_pick_order_has_fixed_dimension():
    """
    A full 20-pick order should still produce exactly
    62 values.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
        max_order_size=20,
    )

    order = all_pick_locations[
        :20
    ]

    state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    encoded_state = encoder.encode(
        state
    )

    assert encoded_state.shape == (
        62,
    )


# ============================================================
# ACTIVE FLAGS
# ============================================================

def test_twenty_pick_order_has_twenty_active_flags():
    """
    Before collection, a 20-pick order should contain exactly
    20 active slot flags.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    order = all_pick_locations[
        :20
    ]

    state = env.reset(
        order
    )

    encoder.reset(
        order
    )

    encoded_state = encoder.encode(
        state
    )

    active_flags = encoded_state[
        4::3
    ]

    assert len(
        active_flags
    ) == 20

    assert np.sum(
        active_flags
    ) == pytest.approx(
        20.0
    )


# ============================================================
# TOO MANY PICKS
# ============================================================

def test_more_than_twenty_picks_is_rejected():
    """
    Orders exceeding the encoder capacity should fail
    explicitly rather than silently truncating the order.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
        max_order_size=20,
    )

    order = all_pick_locations[
        :21
    ]

    with pytest.raises(
        ValueError
    ):

        encoder.reset(
            order
        )


# ============================================================
# INVALID PICK LOCATION
# ============================================================

def test_invalid_pick_location_is_rejected():
    """
    The encoder should reject a location that is not one of
    the warehouse's 96 fixed pick positions.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    invalid_order = [
        (5, 2)
    ]

    with pytest.raises(
        ValueError
    ):

        encoder.reset(
            invalid_order
        )


# ============================================================
# DUPLICATE PICK LOCATION
# ============================================================

def test_duplicate_pick_locations_are_rejected():
    """
    An order should not contain the same storage location
    more than once.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    duplicate_order = [
        (2, 1),
        (2, 1),
    ]

    with pytest.raises(
        ValueError
    ):

        encoder.reset(
            duplicate_order
        )


# ============================================================
# ENCODER MUST RECEIVE ORDER FIRST
# ============================================================

def test_encode_before_setting_order_is_rejected():
    """
    The encoder must know the episode's original order before
    it can assign stable pick slots.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    state = env.reset(
        [
            (8, 7)
        ]
    )

    with pytest.raises(
        RuntimeError
    ):

        encoder.encode(
            state
        )


# ============================================================
# INCORRECT ENVIRONMENT STATE DIMENSION
# ============================================================

def test_incorrect_environment_state_dimension_is_rejected():
    """
    The encoder expects the WarehouseEnv 98-dimensional
    state and should reject incorrectly sized input.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
        env,
    ) = create_test_warehouse()

    encoder = create_encoder(
        all_pick_locations,
        grid,
    )

    encoder.reset(
        [
            (8, 7)
        ]
    )

    incorrect_state = np.zeros(
        10,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        encoder.encode(
            incorrect_state
        )
