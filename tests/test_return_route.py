import numpy as np
import pytest

from heuristics.return_route import return_route


def create_test_warehouse():
    """
    Create the same 14 x 17 warehouse used by the project.
    """

    height = 14
    width = 17

    grid = np.ones((height, width), dtype=np.int8)

    # Front cross aisle
    grid[0, :] = 0

    # Rear cross aisle
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

    distance = return_route(
        grid,
        depot,
        []
    )

    assert distance == 0.0


# ============================================================
# TEST 2: ONE SHALLOW PICK
# ============================================================

def test_one_shallow_pick():
    """
    Pick at (2,1).

    Route:

    depot (0,0) -> aisle 1      = 1
    row 0 -> row 2              = 2
    row 2 -> row 0              = 2
    aisle 1 -> depot            = 1

    Total = 6
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1)
    ]

    distance = return_route(
        grid,
        depot,
        picks
    )

    assert distance == 6.0


# ============================================================
# TEST 3: ONE DEEP PICK
# ============================================================

def test_one_deep_pick():
    """
    Pick at (10,1).

    Route:

    depot -> aisle 1            = 1
    enter to row 10             = 10
    return to front             = 10
    aisle 1 -> depot            = 1

    Total = 22
    """

    grid, depot = create_test_warehouse()

    picks = [
        (10, 1)
    ]

    distance = return_route(
        grid,
        depot,
        picks
    )

    assert distance == 22.0


# ============================================================
# TEST 4: MULTIPLE PICKS IN SAME AISLE
# ============================================================

def test_multiple_picks_same_aisle():
    """
    Required picks:
        (2,1)
        (6,1)
        (11,1)

    Return only needs to travel as far as the
    deepest required pick, row 11.

    Route:

    depot -> aisle 1            = 1
    row 0 -> row 11             = 11
    row 11 -> row 0             = 11
    aisle 1 -> depot            = 1

    Total = 24
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1),
        (6, 1),
        (11, 1)
    ]

    distance = return_route(
        grid,
        depot,
        picks
    )

    assert distance == 24.0


# ============================================================
# TEST 5: TWO REQUIRED AISLES
# ============================================================

def test_two_required_aisles():
    """
    Required picks:
        aisle 1 -> row 4
        aisle 3 -> row 8

    Route:

    depot -> aisle 1            = 1

    aisle 1:
        row 0 -> row 4          = 4
        row 4 -> row 0          = 4

    aisle 1 -> aisle 3          = 2

    aisle 3:
        row 0 -> row 8          = 8
        row 8 -> row 0          = 8

    aisle 3 -> depot            = 3

    Total:

    1 + 8 + 2 + 16 + 3 = 30
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),
        (8, 3)
    ]

    distance = return_route(
        grid,
        depot,
        picks
    )

    assert distance == 30.0


# ============================================================
# TEST 6: THREE REQUIRED AISLES
# ============================================================

def test_three_required_aisles():
    """
    Required picks:
        aisle 1 -> row 3
        aisle 3 -> row 7
        aisle 5 -> row 10

    Horizontal travel:
        depot -> aisle 1 = 1
        aisle 1 -> 3     = 2
        aisle 3 -> 5     = 2
        aisle 5 -> depot = 5

        total horizontal = 10

    Aisle travel:
        aisle 1 = 2 * 3  = 6
        aisle 3 = 2 * 7  = 14
        aisle 5 = 2 * 10 = 20

        total aisle = 40

    Total = 50
    """

    grid, depot = create_test_warehouse()

    picks = [
        (3, 1),
        (7, 3),
        (10, 5)
    ]

    distance = return_route(
        grid,
        depot,
        picks
    )

    assert distance == 50.0


# ============================================================
# TEST 7: DEEPER PICK SHOULD INCREASE DISTANCE
# ============================================================

def test_pick_depth_changes_return_distance():
    """
    Unlike S-shape, Return distance depends on how
    deep the required item is located.
    """

    grid, depot = create_test_warehouse()

    shallow_pick = [
        (2, 1)
    ]

    deep_pick = [
        (10, 1)
    ]

    shallow_distance = return_route(
        grid,
        depot,
        shallow_pick
    )

    deep_distance = return_route(
        grid,
        depot,
        deep_pick
    )

    assert shallow_distance == 6.0
    assert deep_distance == 22.0

    assert deep_distance > shallow_distance


# ============================================================
# TEST 8: ONLY DEEPEST PICK MATTERS WITHIN AN AISLE
# ============================================================

def test_extra_shallow_picks_do_not_change_distance():
    """
    If the deepest pick is unchanged, adding shallower
    picks in the same aisle should not change Return distance.
    """

    grid, depot = create_test_warehouse()

    one_pick = [
        (10, 1)
    ]

    multiple_picks = [
        (2, 1),
        (5, 1),
        (10, 1)
    ]

    distance_one = return_route(
        grid,
        depot,
        one_pick
    )

    distance_multiple = return_route(
        grid,
        depot,
        multiple_picks
    )

    assert distance_one == distance_multiple
    assert distance_one == 22.0


# ============================================================
# TEST 9: INVALID PICK LOCATION
# ============================================================

def test_invalid_pick_location():
    """
    Column 2 is an obstacle/storage block rather than
    one of the valid picking aisle columns.
    """

    grid, depot = create_test_warehouse()

    invalid_picks = [
        (5, 2)
    ]

    with pytest.raises(ValueError):
        return_route(
            grid,
            depot,
            invalid_picks
        )
