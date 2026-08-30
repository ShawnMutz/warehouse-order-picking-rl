"""
Potential-based reward shaping for warehouse routing.

This module provides a routing-aware potential function that can
be used for orders containing between 1 and 20 picks.

The potential does NOT use the exact optimal-routing solver.

Instead, it estimates remaining routing work using:

    1. shortest distance from the picker to any remaining pick,
    2. minimum spanning tree (MST) cost connecting remaining picks,
    3. shortest distance from any remaining pick to the depot.

When picks remain:

    estimate =
        entry_distance
        + MST_cost
        + exit_distance

    Phi(s) = -estimate

When all picks have been collected:

    Phi(s) =
        -distance(current_position, depot)

At successful completion:

    current_position = depot
    no picks remain

therefore:

    Phi(s) = 0

Potential-based shaping is:

    F(s, s') =
        scale * [gamma * Phi(s') - Phi(s)]

The raw WarehouseEnv state is expected:

    [
        normalised_picker_row,
        normalised_picker_col,
        remaining_pick_bit_1,
        ...
        remaining_pick_bit_96
    ]

WarehouseEnv orders the 96 location bits using sorted coordinate
order, so this module uses the same ordering when decoding them.
"""

from collections import deque

import numpy as np


# ============================================================
# SHORTEST-PATH DISTANCES
# ============================================================

def bfs_distance_map(
    grid,
    start,
):
    """
    Calculate shortest-path distances from one walkable
    warehouse position to every reachable walkable position.

    Parameters
    ----------
    grid : np.ndarray
        Warehouse grid where:

            0 = walkable
            1 = obstacle

    start : tuple[int, int]
        Starting coordinate.

    Returns
    -------
    dict
        Mapping:

            coordinate -> shortest-path distance
    """

    grid = np.asarray(
        grid
    )

    if grid.ndim != 2:
        raise ValueError(
            "grid must be a two-dimensional array."
        )

    rows, cols = grid.shape

    start = tuple(
        start
    )

    start_row, start_col = start

    if not (
        0 <= start_row < rows
        and
        0 <= start_col < cols
    ):
        raise ValueError(
            f"Start position is outside the grid: {start}"
        )

    if grid[start] != 0:
        raise ValueError(
            f"Start position is not walkable: {start}"
        )

    distances = {
        start: 0
    }

    queue = deque(
        [start]
    )

    directions = [
        (-1, 0),   # up
        (1, 0),    # down
        (0, -1),   # left
        (0, 1),    # right
    ]

    while queue:

        current = queue.popleft()

        row, col = current

        current_distance = distances[
            current
        ]

        for dr, dc in directions:

            next_row = (
                row
                + dr
            )

            next_col = (
                col
                + dc
            )

            if not (
                0 <= next_row < rows
                and
                0 <= next_col < cols
            ):
                continue

            if grid[
                next_row,
                next_col
            ] != 0:
                continue

            next_position = (
                next_row,
                next_col,
            )

            if next_position in distances:
                continue

            distances[
                next_position
            ] = (
                current_distance
                + 1
            )

            queue.append(
                next_position
            )

    return distances


def build_distance_lookup(
    grid,
    depot,
    all_pick_locations,
):
    """
    Pre-compute shortest-path distance maps from:

        - the depot
        - every fixed warehouse pick location

    This is sufficient for reward shaping because the picker
    position can be looked up inside the BFS map associated
    with a pick or the depot.

    Parameters
    ----------
    grid : np.ndarray
        Warehouse grid.

    depot : tuple[int, int]
        Depot coordinate.

    all_pick_locations : iterable
        All fixed pick locations.

    Returns
    -------
    dict
        Nested distance lookup.
    """

    relevant_points = [
        tuple(
            depot
        ),
        *[
            tuple(
                location
            )
            for location
            in all_pick_locations
        ],
    ]

    # Avoid duplicate BFS calculations.
    relevant_points = list(
        dict.fromkeys(
            relevant_points
        )
    )

    distance_lookup = {}

    for point in relevant_points:

        distance_lookup[
            point
        ] = bfs_distance_map(
            grid,
            point,
        )

    return distance_lookup


# ============================================================
# RAW STATE DECODING
# ============================================================

def decode_agent_position(
    environment_state,
    grid_shape,
):
    """
    Recover the discrete picker coordinate from the first
    two normalised WarehouseEnv state values.

    Parameters
    ----------
    environment_state : array-like
        Raw WarehouseEnv state.

    grid_shape : tuple[int, int]
        Warehouse shape:

            (rows, columns)
    """

    environment_state = np.asarray(
        environment_state,
        dtype=np.float32,
    )

    if environment_state.ndim != 1:

        raise ValueError(
            "environment_state must be one-dimensional."
        )

    if len(
        grid_shape
    ) != 2:

        raise ValueError(
            "grid_shape must contain (rows, columns)."
        )

    rows = int(
        grid_shape[0]
    )

    cols = int(
        grid_shape[1]
    )

    if rows <= 1 or cols <= 1:

        raise ValueError(
            "grid_shape must contain at least two rows "
            "and two columns."
        )

    if environment_state.size < 2:

        raise ValueError(
            "environment_state must contain at least "
            "two picker-coordinate values."
        )

    row = int(
        round(
            float(
                environment_state[0]
            )
            * (
                rows - 1
            )
        )
    )

    col = int(
        round(
            float(
                environment_state[1]
            )
            * (
                cols - 1
            )
        )
    )

    row = max(
        0,
        min(
            rows - 1,
            row,
        ),
    )

    col = max(
        0,
        min(
            cols - 1,
            col,
        ),
    )

    return (
        row,
        col,
    )


def get_remaining_pick_locations(
    environment_state,
    all_pick_locations,
):
    """
    Decode the set of remaining picks from a raw WarehouseEnv
    state.

    WarehouseEnv stores its pick bits in sorted coordinate
    order, so the same ordering must be used here.
    """

    environment_state = np.asarray(
        environment_state,
        dtype=np.float32,
    )

    state_locations = sorted(
        tuple(
            location
        )
        for location
        in all_pick_locations
    )

    expected_state_dim = (
        2
        + len(
            state_locations
        )
    )

    if environment_state.ndim != 1:

        raise ValueError(
            "environment_state must be one-dimensional."
        )

    if len(
        environment_state
    ) != expected_state_dim:

        raise ValueError(
            "Incorrect WarehouseEnv state dimension: "
            f"expected {expected_state_dim}, "
            f"received {len(environment_state)}."
        )

    remaining_bits = (
        environment_state[
            2:
        ]
    )

    remaining_locations = []

    for (
        location,
        bit,
    ) in zip(
        state_locations,
        remaining_bits,
    ):

        if bit > 0.5:

            remaining_locations.append(
                location
            )

    return remaining_locations


# ============================================================
# DISTANCE HELPERS
# ============================================================

def get_distance(
    distance_lookup,
    point_a,
    point_b,
):
    """
    Retrieve a shortest-path distance.

    The pre-computed lookup contains maps starting from the
    depot and fixed pick locations.

    Because warehouse movement distances are symmetric, if
    point_a is not itself a lookup source but point_b is,
    the reverse lookup can be used.
    """

    point_a = tuple(
        point_a
    )

    point_b = tuple(
        point_b
    )

    if (
        point_a in distance_lookup
        and
        point_b in distance_lookup[
            point_a
        ]
    ):

        return float(
            distance_lookup[
                point_a
            ][
                point_b
            ]
        )

    if (
        point_b in distance_lookup
        and
        point_a in distance_lookup[
            point_b
        ]
    ):

        return float(
            distance_lookup[
                point_b
            ][
                point_a
            ]
        )

    raise KeyError(
        "Distance could not be found between "
        f"{point_a} and {point_b}."
    )


# ============================================================
# MINIMUM SPANNING TREE
# ============================================================

def minimum_spanning_tree_cost(
    points,
    distance_lookup,
):
    """
    Calculate the minimum spanning tree cost connecting a
    collection of remaining pick locations.

    Prim's algorithm is used.

    The maximum final order size is only 20, so the simple
    O(n^2) implementation is more than sufficient.

    Parameters
    ----------
    points : iterable[tuple[int, int]]
        Remaining pick locations.

    distance_lookup : dict
        Pre-computed shortest-path lookup.

    Returns
    -------
    float
        Total MST edge cost.

    Notes
    -----
    Zero or one point requires no connecting edges, so the
    MST cost is zero.
    """

    points = [
        tuple(
            point
        )
        for point in points
    ]

    # Remove accidental duplicates while retaining a
    # deterministic ordering.
    points = sorted(
        set(
            points
        )
    )

    if len(
        points
    ) <= 1:

        return 0.0

    visited = {
        points[0]
    }

    unvisited = set(
        points[
            1:
        ]
    )

    total_cost = 0.0

    while unvisited:

        best_cost = None
        best_next_point = None

        for visited_point in visited:

            for unvisited_point in unvisited:

                edge_cost = get_distance(
                    distance_lookup,
                    visited_point,
                    unvisited_point,
                )

                if (
                    best_cost is None
                    or
                    edge_cost < best_cost
                ):

                    best_cost = edge_cost

                    best_next_point = (
                        unvisited_point
                    )

        if best_next_point is None:

            raise RuntimeError(
                "Unable to construct minimum spanning tree."
            )

        total_cost += float(
            best_cost
        )

        visited.add(
            best_next_point
        )

        unvisited.remove(
            best_next_point
        )

    return float(
        total_cost
    )


# ============================================================
# ROUTING POTENTIAL
# ============================================================

def calculate_routing_potential(
    environment_state,
    grid_shape,
    depot,
    all_pick_locations,
    distance_lookup,
):
    """
    Calculate routing-aware state potential.

    If picks remain:

        estimated_remaining_work =
            distance picker -> nearest remaining pick
            +
            MST cost connecting remaining picks
            +
            distance nearest remaining pick -> depot

        Phi(s) =
            -estimated_remaining_work

    More precisely, the entry and exit locations are selected
    independently:

        entry =
            min distance(current, p)

        exit =
            min distance(p, depot)

    over all remaining picks p.

    If no picks remain:

        Phi(s) =
            -distance(current, depot)

    At successful completion the picker is at the depot and
    therefore:

        Phi(s) = 0

    The function uses no exact routing solver and does not
    prescribe a specific visitation sequence.
    """

    environment_state = np.asarray(
        environment_state,
        dtype=np.float32,
    )

    current_position = (
        decode_agent_position(
            environment_state,
            grid_shape,
        )
    )

    remaining_picks = (
        get_remaining_pick_locations(
            environment_state,
            all_pick_locations,
        )
    )

    depot = tuple(
        depot
    )

    # --------------------------------------------------------
    # All picks collected
    # --------------------------------------------------------

    if not remaining_picks:

        distance_to_depot = get_distance(
            distance_lookup,
            current_position,
            depot,
        )

        return -float(
            distance_to_depot
        )

    # --------------------------------------------------------
    # Distance from picker to remaining set
    # --------------------------------------------------------

    entry_distance = min(
        get_distance(
            distance_lookup,
            current_position,
            pick,
        )
        for pick
        in remaining_picks
    )

    # --------------------------------------------------------
    # Connect remaining picks
    # --------------------------------------------------------

    mst_cost = (
        minimum_spanning_tree_cost(
            remaining_picks,
            distance_lookup,
        )
    )

    # --------------------------------------------------------
    # Distance from remaining set back to depot
    # --------------------------------------------------------

    exit_distance = min(
        get_distance(
            distance_lookup,
            pick,
            depot,
        )
        for pick
        in remaining_picks
    )

    estimated_remaining_work = (
        entry_distance
        + mst_cost
        + exit_distance
    )

    return -float(
        estimated_remaining_work
    )


# ============================================================
# POTENTIAL-BASED REWARD SHAPING
# ============================================================

def calculate_potential_shaping_reward(
    environment_state,
    next_environment_state,
    grid_shape,
    depot,
    all_pick_locations,
    distance_lookup,
    gamma=0.99,
    scale=1.0,
):
    """
    Calculate potential-based shaping reward:

        F(s, s') =
            scale *
            [gamma * Phi(s') - Phi(s)]

    Parameters
    ----------
    environment_state
        Raw WarehouseEnv state before the action.

    next_environment_state
        Raw WarehouseEnv state after the action.

    gamma : float
        Discount factor used by the DQN.

    scale : float
        Reward-shaping multiplier.
    """

    gamma = float(
        gamma
    )

    scale = float(
        scale
    )

    if not (
        0.0 <= gamma <= 1.0
    ):

        raise ValueError(
            "gamma must lie between 0 and 1."
        )

    if not np.isfinite(
        scale
    ):

        raise ValueError(
            "scale must be finite."
        )

    current_potential = (
        calculate_routing_potential(
            environment_state,
            grid_shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    next_potential = (
        calculate_routing_potential(
            next_environment_state,
            grid_shape,
            depot,
            all_pick_locations,
            distance_lookup,
        )
    )

    shaping_reward = (
        scale
        * (
            gamma
            * next_potential
            - current_potential
        )
    )

    return float(
        shaping_reward
    )


# ============================================================
# SHORT ALIASES
# ============================================================

calculate_potential = (
    calculate_routing_potential
)

calculate_shaping_reward = (
    calculate_potential_shaping_reward
)
