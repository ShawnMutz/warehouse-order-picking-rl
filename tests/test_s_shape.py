import numpy as np

from heuristics.s_shape import s_shape_route


def create_test_warehouse():
    """
    Create the same 14 x 17 warehouse used by the project.
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


def test_no_picks():
    grid, depot = create_test_warehouse()

    distance = s_shape_route(
        grid,
        depot,
        []
    )

    assert distance == 0.0


def test_one_required_aisle():
    """
    One pick in aisle column 1.

    Expected route:

    depot (0,0)
        -> aisle entrance (0,1)        = 1

    traverse aisle front -> rear       = 13

    return rear -> front same aisle    = 13

    front entrance -> depot            = 1

    Total = 28
    """

    grid, depot = create_test_warehouse()

    picks = [
        (5, 1)
    ]

    distance = s_shape_route(
        grid,
        depot,
        picks
    )

    assert distance == 28.0


def test_multiple_picks_same_aisle():
    """
    S-shape traverses the complete aisle regardless
    of how many items are required within that aisle.

    Therefore this should have the same distance
    as the one-pick case.
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1),
        (6, 1),
        (11, 1)
    ]

    distance = s_shape_route(
        grid,
        depot,
        picks
    )

    assert distance == 28.0


def test_two_required_aisles():
    """
    Required aisles: columns 1 and 3.

    Route:

    depot -> aisle 1                   = 1
    aisle 1 front -> rear              = 13
    rear cross aisle: col 1 -> col 3   = 2
    aisle 3 rear -> front              = 13
    front cross aisle: col 3 -> depot  = 3

    Total:

    1 + 13 + 2 + 13 + 3 = 32
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),
        (8, 3)
    ]

    distance = s_shape_route(
        grid,
        depot,
        picks
    )

    assert distance == 32.0


def test_three_required_aisles():
    """
    Required aisles: columns 1, 3 and 5.

    Route:

    depot -> aisle 1                   = 1
    aisle 1 front -> rear              = 13

    aisle 1 -> aisle 3                 = 2
    aisle 3 rear -> front              = 13

    aisle 3 -> aisle 5                 = 2
    aisle 5 front -> rear              = 13

    Since there are an odd number of required aisles,
    the picker finishes at the rear.

    Return down aisle 5                = 13
    front aisle 5 -> depot             = 5

    Total:

    1 + 13 + 2 + 13 + 2 + 13 + 13 + 5
    = 62
    """

    grid, depot = create_test_warehouse()

    picks = [
        (3, 1),
        (7, 3),
        (10, 5)
    ]

    distance = s_shape_route(
        grid,
        depot,
        picks
    )

    assert distance == 62.0


def test_pick_depth_does_not_change_s_shape_distance():
    """
    S-shape fully traverses every required aisle.

    A pick near the front and a pick near the rear
    should therefore produce the same route distance
    if they are in the same aisle.
    """

    grid, depot = create_test_warehouse()

    front_pick = [
        (1, 1)
    ]

    rear_pick = [
        (12, 1)
    ]

    front_distance = s_shape_route(
        grid,
        depot,
        front_pick
    )

    rear_distance = s_shape_route(
        grid,
        depot,
        rear_pick
    )

    assert front_distance == rear_distance
    assert front_distance == 28.0
