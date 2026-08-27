import random

import numpy as np
import pytest

from exact.held_karp_validator import held_karp_distance
from exact.ratliff_rosenthal import (
    ratliff_rosenthal_distance,
    exact_optimal_distance,
)


# ============================================================
# TEST WAREHOUSE
# ============================================================

def create_test_warehouse():
    """
    Create the project's 14 x 17 single-block warehouse.
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

    # Eight vertical picking aisles
    aisle_columns = [
        1, 3, 5, 7,
        9, 11, 13, 15
    ]

    for col in aisle_columns:
        grid[:, col] = 0

    depot = (0, 0)

    return grid, depot


def get_all_pick_locations():
    """
    Return the 96 valid storage/pick locations.
    """

    aisle_columns = [
        1, 3, 5, 7,
        9, 11, 13, 15
    ]

    return [
        (row, col)
        for col in aisle_columns
        for row in range(1, 13)
    ]


# ============================================================
# BASIC HAND-CALCULATED TESTS
# ============================================================

def test_no_picks():
    """
    No required items.

    Optimal distance = 0.
    """

    grid, depot = create_test_warehouse()

    distance = ratliff_rosenthal_distance(
        grid,
        depot,
        []
    )

    assert distance == 0.0


def test_one_shallow_pick():
    """
    Pick = (2, 1)

    Depot -> pick = 3
    Pick -> depot = 3

    Optimal distance = 6.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1)
    ]

    rr_distance = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    assert rr_distance == 6.0


def test_one_deep_pick():
    """
    Pick = (10, 1)

    Depot -> pick = 11
    Return = 11

    Optimal distance = 22.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (10, 1)
    ]

    rr_distance = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    assert rr_distance == 22.0


def test_two_picks_same_aisle():
    """
    Picks:
        (2, 1)
        (10, 1)

    Reaching row 10 naturally passes row 2.

    Optimal distance = 22.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1),
        (10, 1)
    ]

    rr_distance = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    assert rr_distance == 22.0


def test_two_picks_different_aisles():
    """
    Picks:
        (4, 1)
        (8, 3)

    Both can be serviced using front-return movements.

    Horizontal travel = 6
    Vertical travel = 8 + 16 = 24

    Optimal distance = 30.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),
        (8, 3)
    ]

    rr_distance = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    assert rr_distance == 30.0


# ============================================================
# COMPARISON WITH HELD-KARP
# ============================================================

@pytest.mark.parametrize(
    "picks",
    [
        [(2, 1)],
        [(10, 1)],
        [(2, 1), (10, 1)],
        [(4, 1), (8, 3)],
        [(2, 1), (5, 3), (9, 5)],
        [(3, 1), (10, 3), (6, 7)],
        [(2, 1), (6, 3), (11, 5), (4, 7)],
        [(3, 1), (8, 3), (5, 5), (10, 7), (2, 9)],
    ]
)
def test_matches_held_karp_known_orders(picks):
    """
    Ratliff-Rosenthal and Held-Karp are independent
    exact methods.

    For small instances they must return the same
    optimal distance.
    """

    grid, depot = create_test_warehouse()

    rr_distance = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    hk_distance = held_karp_distance(
        grid,
        depot,
        picks
    )

    assert rr_distance == pytest.approx(
        hk_distance
    )


# ============================================================
# RANDOM CROSS-VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "seed",
    range(20)
)
def test_matches_held_karp_random_five_pick_orders(seed):
    """
    Cross-validate Ratliff-Rosenthal against Held-Karp
    on deterministic random five-pick orders.

    Five picks keeps Held-Karp computationally cheap while
    providing substantially more varied route structures.
    """

    grid, depot = create_test_warehouse()

    all_pick_locations = get_all_pick_locations()

    rng = random.Random(seed)

    picks = rng.sample(
        all_pick_locations,
        5
    )

    rr_distance = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    hk_distance = held_karp_distance(
        grid,
        depot,
        picks
    )

    assert rr_distance == pytest.approx(
        hk_distance
    ), (
        f"Mismatch for seed {seed}\n"
        f"Picks: {picks}\n"
        f"Ratliff-Rosenthal: {rr_distance}\n"
        f"Held-Karp: {hk_distance}"
    )


# ============================================================
# INPUT ORDER SHOULD NOT MATTER
# ============================================================

def test_pick_input_order_does_not_change_result():
    """
    Rearranging the input list must not change the
    optimal route distance.
    """

    grid, depot = create_test_warehouse()

    picks_a = [
        (2, 1),
        (9, 3),
        (5, 7),
        (11, 9),
        (4, 13)
    ]

    picks_b = list(
        reversed(picks_a)
    )

    distance_a = ratliff_rosenthal_distance(
        grid,
        depot,
        picks_a
    )

    distance_b = ratliff_rosenthal_distance(
        grid,
        depot,
        picks_b
    )

    assert distance_a == pytest.approx(
        distance_b
    )


# ============================================================
# ALIAS TEST
# ============================================================

def test_exact_optimal_distance_alias():
    """
    exact_optimal_distance() should expose the same
    result as ratliff_rosenthal_distance().
    """

    grid, depot = create_test_warehouse()

    picks = [
        (3, 1),
        (8, 3),
        (5, 5)
    ]

    rr_distance = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    alias_distance = exact_optimal_distance(
        grid,
        depot,
        picks
    )

    assert alias_distance == pytest.approx(
        rr_distance
    )


# ============================================================
# VALIDATION TESTS
# ============================================================

def test_duplicate_pick_rejected():
    """
    Duplicate storage locations should not be accepted.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (5, 1),
        (5, 1)
    ]

    with pytest.raises(ValueError):
        ratliff_rosenthal_distance(
            grid,
            depot,
            picks
        )


def test_invalid_aisle_column_rejected():
    """
    Column 2 is an obstacle/storage-block column,
    not a picking aisle.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (5, 2)
    ]

    with pytest.raises(ValueError):
        ratliff_rosenthal_distance(
            grid,
            depot,
            picks
        )


def test_cross_aisle_pick_rejected():
    """
    A pick cannot be placed directly on a cross aisle.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (0, 1)
    ]

    with pytest.raises(ValueError):
        ratliff_rosenthal_distance(
            grid,
            depot,
            picks
        )


def test_invalid_depot_rejected():
    """
    This implementation requires the depot to be
    on the front cross aisle.
    """

    grid, _ = create_test_warehouse()

    invalid_depot = (
        13,
        0
    )

    picks = [
        (5, 1)
    ]

    with pytest.raises(ValueError):
        ratliff_rosenthal_distance(
            grid,
            invalid_depot,
            picks
        )
