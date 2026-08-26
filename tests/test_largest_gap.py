import numpy as np
import pytest

from heuristics.largest_gap import largest_gap_route


def create_test_warehouse():
    """
    Create the project's 14 x 17 single-block warehouse.
    """

    height = 14
    width = 17

    grid = np.ones((height, width), dtype=np.int8)

    # Front and rear cross aisles
    grid[0, :] = 0
    grid[13, :] = 0

    # Eight vertical picking aisles
    aisle_columns = [1, 3, 5, 7, 9, 11, 13, 15]

    for col in aisle_columns:
        grid[:, col] = 0

    depot = (0, 0)

    return grid, depot


# ============================================================
# TEST 1: NO PICKS
# ============================================================

def test_no_picks():
    grid, depot = create_test_warehouse()

    distance = largest_gap_route(
        grid,
        depot,
        []
    )

    # Expected: 0.0
    assert distance == 0.0


# ============================================================
# TEST 2: ONE SHALLOW PICK
# ============================================================

def test_one_shallow_pick():
    """
    One required aisle uses Return behaviour.

    depot -> aisle 1 = 1
    row 0 -> row 2   = 2
    row 2 -> row 0   = 2
    aisle 1 -> depot = 1

    Total = 6
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1)
    ]

    distance = largest_gap_route(
        grid,
        depot,
        picks
    )

    # Expected: 6.0
    assert distance == 6.0


# ============================================================
# TEST 3: ONE DEEP PICK
# ============================================================

def test_one_deep_pick():
    """
    depot -> aisle 1 = 1
    enter to row 10  = 10
    return to front  = 10
    aisle 1 -> depot = 1

    Total = 22
    """

    grid, depot = create_test_warehouse()

    picks = [
        (10, 1)
    ]

    distance = largest_gap_route(
        grid,
        depot,
        picks
    )

    # Expected: 22.0
    assert distance == 22.0


# ============================================================
# TEST 4: TWO REQUIRED AISLES
# ============================================================

def test_two_required_aisles():
    """
    With two required aisles, both are traversed completely.

    Picks:
        aisle 1 -> row 4
        aisle 3 -> row 8

    Horizontal:
        depot -> aisle 1 = 1
        aisle 1 -> aisle 3 = 2
        aisle 3 -> depot = 3

        total horizontal = 6

    Vertical:
        aisle 1 full traversal = 13
        aisle 3 full traversal = 13

        total vertical = 26

    Total = 32
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),
        (8, 3)
    ]

    distance = largest_gap_route(
        grid,
        depot,
        picks
    )

    # Expected: 32.0
    assert distance == 32.0


# ============================================================
# TEST 5: LARGEST GAP BETWEEN PICKS
# ============================================================

def test_middle_largest_gap():
    """
    Required aisle columns:
        1, 3, 5

    Middle aisle 3 contains:
        rows 2, 5, 10

    Points:
        [0, 2, 5, 10, 13]

    Gaps:
        [2, 3, 5, 3]

    Largest gap = 5

    Middle aisle travel:
        2 * (13 - 5) = 16

    Horizontal:
        1 + 4 + 5 = 10

    First + last full aisle traversals:
        13 + 13 = 26

    Total:
        10 + 26 + 16 = 52
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),

        (2, 3),
        (5, 3),
        (10, 3),

        (8, 5)
    ]

    distance = largest_gap_route(
        grid,
        depot,
        picks
    )

    # Expected: 52.0
    assert distance == 52.0


# ============================================================
# TEST 6: LARGEST GAP AT FRONT
# ============================================================

def test_largest_gap_at_front():
    """
    Middle aisle 3:
        rows = [8, 10]

    Points:
        [0, 8, 10, 13]

    Gaps:
        [8, 2, 3]

    Largest gap = 8

    Middle aisle travel:
        2 * (13 - 8) = 10

    Horizontal = 10
    First + last full traversals = 26

    Total = 46
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),

        (8, 3),
        (10, 3),

        (6, 5)
    ]

    distance = largest_gap_route(
        grid,
        depot,
        picks
    )

    # Expected: 46.0
    assert distance == 46.0


# ============================================================
# TEST 7: LARGEST GAP AT REAR
# ============================================================

def test_largest_gap_at_rear():
    """
    Middle aisle 3:
        rows = [2, 4]

    Points:
        [0, 2, 4, 13]

    Gaps:
        [2, 2, 9]

    Largest gap = 9

    Middle aisle travel:
        2 * (13 - 9) = 8

    Horizontal = 10
    First + last full traversals = 26

    Total = 44
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),

        (2, 3),
        (4, 3),

        (6, 5)
    ]

    distance = largest_gap_route(
        grid,
        depot,
        picks
    )

    # Expected: 44.0
    assert distance == 44.0


# ============================================================
# TEST 8: MULTIPLE INTERMEDIATE AISLES
# ============================================================

def test_multiple_middle_aisles():
    """
    Required aisles:
        1, 3, 5, 7

    Aisle 3:
        points = [0, 2, 5, 10, 13]
        gaps = [2, 3, 5, 3]
        largest gap = 5
        vertical = 2 * (13 - 5) = 16

    Aisle 5:
        points = [0, 3, 9, 13]
        gaps = [3, 6, 4]
        largest gap = 6
        vertical = 2 * (13 - 6) = 14

    Horizontal:
        depot -> 1 = 1
        aisle 1 -> aisle 7 = 6
        aisle 7 -> depot = 7

        total = 14

    First + last full traversals:
        26

    Middle aisle travel:
        16 + 14 = 30

    Total:
        14 + 26 + 30 = 70
    """

    grid, depot = create_test_warehouse()

    picks = [
        (5, 1),

        (2, 3),
        (5, 3),
        (10, 3),

        (3, 5),
        (9, 5),

        (6, 7)
    ]

    distance = largest_gap_route(
        grid,
        depot,
        picks
    )

    # Expected: 70.0
    assert distance == 70.0


# ============================================================
# TEST 9: PICK DISTRIBUTION SHOULD MATTER
# ============================================================

def test_middle_pick_distribution_changes_distance():
    """
    Largest Gap should respond to the spatial distribution
    of picks in an intermediate aisle.
    """

    grid, depot = create_test_warehouse()

    order_a = [
        (5, 1),

        (2, 3),
        (5, 3),
        (10, 3),

        (5, 5)
    ]

    order_b = [
        (5, 1),

        (5, 3),
        (6, 3),
        (7, 3),

        (5, 5)
    ]

    distance_a = largest_gap_route(
        grid,
        depot,
        order_a
    )

    distance_b = largest_gap_route(
        grid,
        depot,
        order_b
    )

    assert distance_a == 52.0
    assert distance_b == 50.0

    assert distance_a != distance_b

# ============================================================
# TEST 10: INVALID PICK LOCATION
# ============================================================


def test_invalid_pick_location():
    """
    Column 2 is not a picking aisle.
    """

    grid, depot = create_test_warehouse()

    invalid_picks = [
        (5, 2)
    ]

    with pytest.raises(ValueError):
        largest_gap_route(
            grid,
            depot,
            invalid_picks
        )
