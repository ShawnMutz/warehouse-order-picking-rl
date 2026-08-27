import numpy as np
import pytest

from exact.held_karp_validator import (
    bfs_shortest_distances,
    pairwise_distances,
    held_karp_distance,
)


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
# BFS TESTS
# ============================================================

def test_bfs_start_distance_is_zero():
    grid, depot = create_test_warehouse()

    distances = bfs_shortest_distances(
        grid,
        depot
    )

    assert distances[depot] == 0.0


def test_bfs_distance_to_shallow_pick():
    """
    Depot = (0, 0)
    Pick = (2, 1)

    Route:
        right 1
        down 2

    Distance = 3
    """

    grid, depot = create_test_warehouse()

    distances = bfs_shortest_distances(
        grid,
        depot
    )

    assert distances[2, 1] == 3.0


def test_bfs_distance_to_deep_pick():
    """
    Depot = (0, 0)
    Pick = (10, 1)

    Route:
        right 1
        down 10

    Distance = 11
    """

    grid, depot = create_test_warehouse()

    distances = bfs_shortest_distances(
        grid,
        depot
    )

    assert distances[10, 1] == 11.0


def test_bfs_obstacle_is_unreachable():
    """
    (5, 2) is an obstacle because column 2 is not
    a picking aisle and row 5 is not a cross aisle.
    """

    grid, depot = create_test_warehouse()

    distances = bfs_shortest_distances(
        grid,
        depot
    )

    assert np.isinf(distances[5, 2])


# ============================================================
# PAIRWISE DISTANCE TESTS
# ============================================================

def test_pairwise_matrix_shape():
    grid, depot = create_test_warehouse()

    points = [
        depot,
        (2, 1),
        (10, 1)
    ]

    distances = pairwise_distances(
        grid,
        points
    )

    assert distances.shape == (3, 3)


def test_pairwise_diagonal_is_zero():
    grid, depot = create_test_warehouse()

    points = [
        depot,
        (2, 1),
        (10, 1)
    ]

    distances = pairwise_distances(
        grid,
        points
    )

    assert distances[0, 0] == 0.0
    assert distances[1, 1] == 0.0
    assert distances[2, 2] == 0.0


def test_pairwise_distances_are_symmetric():
    grid, depot = create_test_warehouse()

    points = [
        depot,
        (2, 1),
        (10, 3)
    ]

    distances = pairwise_distances(
        grid,
        points
    )

    assert np.array_equal(
        distances,
        distances.T
    )


def test_pairwise_known_distances():
    """
    Points:
        depot  = (0, 0)
        pick A = (2, 1)
        pick B = (10, 1)

    Expected:
        depot -> A = 3
        depot -> B = 11
        A -> B = 8
    """

    grid, depot = create_test_warehouse()

    points = [
        depot,
        (2, 1),
        (10, 1)
    ]

    distances = pairwise_distances(
        grid,
        points
    )

    assert distances[0, 1] == 3.0
    assert distances[0, 2] == 11.0
    assert distances[1, 2] == 8.0


# ============================================================
# HELD-KARP TESTS
# ============================================================

def test_no_picks():
    """
    No picks means the picker does not leave the depot.

    Optimal distance = 0.
    """

    grid, depot = create_test_warehouse()

    distance = held_karp_distance(
        grid,
        depot,
        []
    )

    assert distance == 0.0


def test_one_shallow_pick():
    """
    Required pick = (2, 1)

    Depot -> pick:
        1 horizontal + 2 vertical = 3

    Return:
        3

    Total = 6
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1)
    ]

    distance = held_karp_distance(
        grid,
        depot,
        picks
    )

    assert distance == 6.0


def test_one_deep_pick():
    """
    Required pick = (10, 1)

    Depot -> pick = 11
    Pick -> depot = 11

    Total = 22
    """

    grid, depot = create_test_warehouse()

    picks = [
        (10, 1)
    ]

    distance = held_karp_distance(
        grid,
        depot,
        picks
    )

    assert distance == 22.0


def test_two_picks_same_aisle():
    """
    Picks:
        (2, 1)
        (10, 1)

    Optimal route reaches the deepest pick while
    collecting the shallow pick on the way.

    Total = 22
    """

    grid, depot = create_test_warehouse()

    picks = [
        (2, 1),
        (10, 1)
    ]

    distance = held_karp_distance(
        grid,
        depot,
        picks
    )

    assert distance == 22.0


def test_two_different_aisles():
    """
    Picks:
        A = (4, 1)
        B = (8, 3)

    Distances:
        depot -> A = 5
        A -> B = 14
        B -> depot = 11

    Total = 30
    """

    grid, depot = create_test_warehouse()

    picks = [
        (4, 1),
        (8, 3)
    ]

    distance = held_karp_distance(
        grid,
        depot,
        picks
    )

    assert distance == 30.0


def test_pick_order_does_not_change_optimum():
    """
    Held-Karp should return the same optimum regardless
    of the order in which required picks are supplied.
    """

    grid, depot = create_test_warehouse()

    picks_a = [
        (2, 1),
        (10, 1),
        (5, 3)
    ]

    picks_b = [
        (5, 3),
        (2, 1),
        (10, 1)
    ]

    distance_a = held_karp_distance(
        grid,
        depot,
        picks_a
    )

    distance_b = held_karp_distance(
        grid,
        depot,
        picks_b
    )

    assert distance_a == distance_b
