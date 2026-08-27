import numpy as np
import pytest

from exact.ratliff_rosenthal import ratliff_rosenthal_distance

from heuristics.s_shape import s_shape_route
from heuristics.return_route import return_route
from heuristics.largest_gap import largest_gap_route

from utils.order_generation import generate_order


# ============================================================
# WAREHOUSE
# ============================================================

def create_test_warehouse():
    """
    Create the project's fixed 14 x 17 single-block warehouse.

    Layout:
        - front cross aisle: row 0
        - rear cross aisle: row 13
        - eight picking aisles
        - depot: (0, 0)
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

    # 12 pick positions per aisle
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
# HELPER
# ============================================================

def calculate_all_distances(
    grid,
    depot,
    picks,
):
    """
    Run the same order through the exact solver and
    all three heuristic routing methods.

    Returns
    -------
    dict
        Distance produced by each method.
    """

    exact = ratliff_rosenthal_distance(
        grid,
        depot,
        picks
    )

    s_shape = s_shape_route(
        grid,
        depot,
        picks
    )

    return_distance = return_route(
        grid,
        depot,
        picks
    )

    largest_gap = largest_gap_route(
        grid,
        depot,
        picks
    )

    return {
        "exact": exact,
        "s_shape": s_shape,
        "return": return_distance,
        "largest_gap": largest_gap,
    }


# ============================================================
# BASIC INTEGRATION TEST
# ============================================================

def test_all_methods_run_on_same_order():
    """
    All four routing methods should successfully process
    the same valid warehouse order.
    """

    (
        grid,
        depot,
        _,
        _,
    ) = create_test_warehouse()

    picks = [
        (2, 1),
        (8, 3),
        (5, 5),
        (10, 7),
        (4, 9),
    ]

    distances = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    assert distances["exact"] >= 0.0
    assert distances["s_shape"] >= 0.0
    assert distances["return"] >= 0.0
    assert distances["largest_gap"] >= 0.0


# ============================================================
# EXACT MUST NOT BE WORSE THAN HEURISTICS
# ============================================================

def test_exact_not_worse_than_heuristics_known_order():
    """
    An exact optimal solution cannot have greater travel
    distance than a feasible heuristic route.
    """

    (
        grid,
        depot,
        _,
        _,
    ) = create_test_warehouse()

    picks = [
        (2, 1),
        (8, 3),
        (5, 5),
        (10, 7),
        (4, 9),
    ]

    distances = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    optimal = distances["exact"]

    assert optimal <= distances["s_shape"]
    assert optimal <= distances["return"]
    assert optimal <= distances["largest_gap"]


# ============================================================
# UNIFORM ORDERS
# ============================================================

@pytest.mark.parametrize(
    "seed",
    range(10),
)
def test_exact_not_worse_uniform_five_pick_orders(seed):
    """
    Generate deterministic random uniform five-pick orders.

    For every order:

        exact <= S-shape
        exact <= Return
        exact <= Largest Gap
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    ) = create_test_warehouse()

    picks = generate_order(
        all_pick_locations=all_pick_locations,
        size=5,
        distribution="uniform",
        seed=seed,
        aisle_columns=aisle_columns,
    )

    distances = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    optimal = distances["exact"]

    assert optimal <= distances["s_shape"], (
        f"S-shape beat exact solver.\n"
        f"Seed: {seed}\n"
        f"Picks: {picks}\n"
        f"Exact: {optimal}\n"
        f"S-shape: {distances['s_shape']}"
    )

    assert optimal <= distances["return"], (
        f"Return beat exact solver.\n"
        f"Seed: {seed}\n"
        f"Picks: {picks}\n"
        f"Exact: {optimal}\n"
        f"Return: {distances['return']}"
    )

    assert optimal <= distances["largest_gap"], (
        f"Largest Gap beat exact solver.\n"
        f"Seed: {seed}\n"
        f"Picks: {picks}\n"
        f"Exact: {optimal}\n"
        f"Largest Gap: {distances['largest_gap']}"
    )


# ============================================================
# CLUSTERED ORDERS
# ============================================================

@pytest.mark.parametrize(
    "seed",
    range(10),
)
def test_exact_not_worse_clustered_five_pick_orders(seed):
    """
    Repeat the integration test using the project's
    clustered OOD order generator.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    ) = create_test_warehouse()

    picks = generate_order(
        all_pick_locations=all_pick_locations,
        size=5,
        distribution="clustered",
        seed=seed,
        aisle_columns=aisle_columns,
    )

    distances = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    optimal = distances["exact"]

    assert optimal <= distances["s_shape"], (
        f"S-shape beat exact solver.\n"
        f"Seed: {seed}\n"
        f"Picks: {picks}\n"
        f"Exact: {optimal}\n"
        f"S-shape: {distances['s_shape']}"
    )

    assert optimal <= distances["return"], (
        f"Return beat exact solver.\n"
        f"Seed: {seed}\n"
        f"Picks: {picks}\n"
        f"Exact: {optimal}\n"
        f"Return: {distances['return']}"
    )

    assert optimal <= distances["largest_gap"], (
        f"Largest Gap beat exact solver.\n"
        f"Seed: {seed}\n"
        f"Picks: {picks}\n"
        f"Exact: {optimal}\n"
        f"Largest Gap: {distances['largest_gap']}"
    )


# ============================================================
# ALL FINAL ORDER SIZES
# ============================================================

@pytest.mark.parametrize(
    "order_size",
    [5, 10, 15, 20],
)
@pytest.mark.parametrize(
    "distribution",
    ["uniform", "clustered"],
)
def test_all_final_experimental_conditions(
    order_size,
    distribution,
):
    """
    Ensure that every routing method can process orders
    corresponding to all eight final experimental conditions:

        4 order sizes x 2 spatial distributions.
    """

    (
        grid,
        depot,
        aisle_columns,
        all_pick_locations,
    ) = create_test_warehouse()

    picks = generate_order(
        all_pick_locations=all_pick_locations,
        size=order_size,
        distribution=distribution,
        seed=42,
        aisle_columns=aisle_columns,
    )

    assert len(picks) == order_size
    assert len(set(picks)) == order_size

    distances = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    optimal = distances["exact"]

    assert np.isfinite(optimal)
    assert np.isfinite(distances["s_shape"])
    assert np.isfinite(distances["return"])
    assert np.isfinite(distances["largest_gap"])

    assert optimal <= distances["s_shape"]
    assert optimal <= distances["return"]
    assert optimal <= distances["largest_gap"]


# ============================================================
# DISTANCE TYPES
# ============================================================

def test_all_methods_return_numeric_distances():
    """
    All routing implementations should return usable
    numeric distance values.
    """

    (
        grid,
        depot,
        _,
        _,
    ) = create_test_warehouse()

    picks = [
        (3, 1),
        (8, 3),
        (5, 7),
        (11, 9),
        (6, 13),
    ]

    distances = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    for method, distance in distances.items():

        assert isinstance(
            distance,
            (int, float, np.integer, np.floating),
        ), (
            f"{method} returned non-numeric "
            f"value: {distance}"
        )

        assert np.isfinite(distance)

        assert distance >= 0.0


# ============================================================
# DETERMINISM
# ============================================================

def test_routing_methods_are_deterministic():
    """
    Running the same order twice should produce identical
    distances for deterministic exact/heuristic methods.
    """

    (
        grid,
        depot,
        _,
        _,
    ) = create_test_warehouse()

    picks = [
        (2, 1),
        (10, 3),
        (4, 5),
        (9, 9),
        (6, 15),
    ]

    first_run = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    second_run = calculate_all_distances(
        grid,
        depot,
        picks,
    )

    assert first_run == second_run
