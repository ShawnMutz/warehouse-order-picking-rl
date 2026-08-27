import numpy as np


# ============================================================
# DEGREE-STATUS CONSTANTS
# ============================================================
#
# Ratliff-Rosenthal distinguishes between:
#
#   0 = zero degree
#   E = positive even degree
#   U = odd degree
#
# Zero and positive-even must be treated separately because
# they represent different partial-route structures.

ZERO = "0"
EVEN = "E"
ODD = "U"


# ============================================================
# INTER-AISLE CONNECTION TYPES
# ============================================================
#
# Each tuple represents:
#
#   (number of front cross-aisle edges,
#    number of rear cross-aisle edges)
#
# Edge multiplicity matters because two traversals produce
# positive-even degree rather than zero degree.

HORIZONTAL_CONFIGS = [
    (0, 0),
    (2, 0),
    (0, 2),
    (1, 1),
    (2, 2),
]


# ============================================================
# WAREHOUSE GEOMETRY
# ============================================================

def get_warehouse_geometry(grid):
    """
    Extract the front/rear cross aisles and vertical picking
    aisle columns from the warehouse grid.

    Parameters
    ----------
    grid : np.ndarray
        Binary warehouse grid:
            0 = traversable
            1 = obstacle

    Returns
    -------
    tuple
        (front_row, rear_row, aisle_columns)
    """

    if not isinstance(grid, np.ndarray):
        raise TypeError("grid must be a NumPy array.")

    if grid.ndim != 2:
        raise ValueError("grid must be two-dimensional.")

    height, width = grid.shape

    cross_rows = [
        row
        for row in range(height)
        if np.all(grid[row, :] == 0)
    ]

    aisle_columns = [
        col
        for col in range(width)
        if np.all(grid[:, col] == 0)
    ]

    if len(cross_rows) < 2:
        raise ValueError(
            "Warehouse must contain a front and rear cross aisle."
        )

    if not aisle_columns:
        raise ValueError(
            "Warehouse must contain at least one vertical picking aisle."
        )

    front_row = min(cross_rows)
    rear_row = max(cross_rows)

    return front_row, rear_row, aisle_columns


# ============================================================
# PICK PREPROCESSING
# ============================================================

def group_picks_by_aisle(picks, aisle_columns):
    """
    Group required pick locations by aisle column.

    Parameters
    ----------
    picks : iterable
        Pick coordinates as (row, column).

    aisle_columns : iterable
        Valid picking-aisle columns.

    Returns
    -------
    dict
        Mapping:
            aisle column -> sorted list of pick rows
    """

    aisle_columns = set(aisle_columns)

    groups = {}

    for row, col in picks:

        if col not in aisle_columns:
            raise ValueError(
                f"Pick ({row}, {col}) is not in a valid picking aisle."
            )

        groups.setdefault(
            col,
            []
        ).append(row)

    for col in groups:
        groups[col].sort()

    return groups


# ============================================================
# DEGREE STATUS HELPERS
# ============================================================

def _edge_status(edge_count):
    """
    Convert an edge multiplicity into a degree-status change.

    0 edges -> ZERO
    odd number -> ODD
    positive even number -> EVEN
    """

    if edge_count == 0:
        return ZERO

    if edge_count % 2 == 0:
        return EVEN

    return ODD


def _combine_degree_status(status_a, status_b):
    """
    Combine two degree contributions.

    Only parity and whether the resulting degree is zero or
    positive are required.

    Examples
    --------
    0 + E -> E
    0 + U -> U
    E + E -> E
    U + U -> E
    E + U -> U
    """

    if status_a == ZERO:
        return status_b

    if status_b == ZERO:
        return status_a

    if status_a == EVEN and status_b == EVEN:
        return EVEN

    if status_a == ODD and status_b == ODD:
        return EVEN

    return ODD


# ============================================================
# INTRA-AISLE CONFIGURATIONS
# ============================================================

def aisle_service_configs(sorted_rows, front_row, rear_row):
    """
    Generate non-dominated vertical service configurations for
    one aisle.

    Each returned configuration is:

        (
            name,
            cost,
            front_degree_addition,
            rear_degree_addition,
            connects_front_to_rear
        )

    The configurations correspond to the relevant ways of
    representing an aisle in the Ratliff-Rosenthal frontier DP.

    For an aisle containing picks, possible configurations are:

        1. Single full traversal
        2. Double full traversal
        3. Return from the front
        4. Return from the rear
        5. Split service around an internal gap

    For an empty aisle:

        1. No vertical travel
        2. Single full traversal
        3. Double full traversal
    """

    rows = sorted(sorted_rows)

    aisle_length = rear_row - front_row

    configs = []

    # --------------------------------------------------------
    # Empty aisle
    # --------------------------------------------------------

    if not rows:

        configs.append(
            (
                "none",
                0.0,
                ZERO,
                ZERO,
                False,
            )
        )

        configs.append(
            (
                "full_single",
                float(aisle_length),
                ODD,
                ODD,
                True,
            )
        )

        configs.append(
            (
                "full_double",
                float(2 * aisle_length),
                EVEN,
                EVEN,
                True,
            )
        )

        return configs

    # --------------------------------------------------------
    # 1. Single full traversal
    # --------------------------------------------------------
    #
    # One traversal connects front and rear.
    # Each endpoint receives degree 1.

    configs.append(
        (
            "full_single",
            float(aisle_length),
            ODD,
            ODD,
            True,
        )
    )

    # --------------------------------------------------------
    # 2. Double full traversal
    # --------------------------------------------------------
    #
    # Two traversals connect front and rear and give each
    # endpoint positive-even degree.

    configs.append(
        (
            "full_double",
            float(2 * aisle_length),
            EVEN,
            EVEN,
            True,
        )
    )

    # --------------------------------------------------------
    # 3. Return from front
    # --------------------------------------------------------

    deepest_pick = max(rows)

    front_return_cost = 2 * (
        deepest_pick - front_row
    )

    configs.append(
        (
            "front_return",
            float(front_return_cost),
            EVEN,
            ZERO,
            False,
        )
    )

    # --------------------------------------------------------
    # 4. Return from rear
    # --------------------------------------------------------

    shallowest_pick = min(rows)

    rear_return_cost = 2 * (
        rear_row - shallowest_pick
    )

    configs.append(
        (
            "rear_return",
            float(rear_return_cost),
            ZERO,
            EVEN,
            False,
        )
    )

    # --------------------------------------------------------
    # 5. Split service
    # --------------------------------------------------------
    #
    # Picks above the chosen gap are serviced from the front.
    # Picks below it are serviced from the rear.
    #
    # Only internal gaps between required picks are relevant.
    # Gaps from a cross aisle to the first/last pick are already
    # represented by the front-return and rear-return cases.

    if len(rows) >= 2:

        internal_gaps = [
            rows[i + 1] - rows[i]
            for i in range(len(rows) - 1)
        ]

        largest_internal_gap = max(
            internal_gaps
        )

        split_cost = 2 * (
            aisle_length - largest_internal_gap
        )

        configs.append(
            (
                "split",
                float(split_cost),
                EVEN,
                EVEN,
                False,
            )
        )

    # --------------------------------------------------------
    # Remove dominated duplicates
    # --------------------------------------------------------
    #
    # Two configurations with identical boundary behaviour are
    # equivalent to the remaining DP. Keep only the cheapest.

    unique = {}

    for config in configs:

        (
            name,
            cost,
            front_add,
            rear_add,
            connects,
        ) = config

        signature = (
            front_add,
            rear_add,
            connects,
        )

        if (
            signature not in unique
            or cost < unique[signature][1]
        ):
            unique[signature] = config

    return list(
        unique.values()
    )


# ============================================================
# VERTICAL TRANSITION
# ============================================================

def _apply_vertical_config(state, config):
    """
    Add one aisle's vertical configuration to a DP state.

    State format
    ------------
    (
        front_degree_status,
        rear_degree_status,
        component_count
    )

    component_count:
        0 = no active component
        1 = one connected component
        2 = two separate components
    """

    (
        front_status,
        rear_status,
        component_count,
    ) = state

    (
        _,
        vertical_cost,
        front_add,
        rear_add,
        connects_front_rear,
    ) = config

    new_front_status = _combine_degree_status(
        front_status,
        front_add,
    )

    new_rear_status = _combine_degree_status(
        rear_status,
        rear_add,
    )

    # --------------------------------------------------------
    # Track connectivity of the two frontier vertices.
    # --------------------------------------------------------

    parent = [0, 1]

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    # Existing frontier endpoints are already connected.
    if (
        component_count == 1
        and front_status != ZERO
        and rear_status != ZERO
    ):
        union(0, 1)

    # Vertical full traversal connects the new front/rear
    # endpoints directly.
    if (
        connects_front_rear
        and front_add != ZERO
        and rear_add != ZERO
    ):
        union(0, 1)

    front_active = (
        new_front_status != ZERO
    )

    rear_active = (
        new_rear_status != ZERO
    )

    if not front_active and not rear_active:

        new_component_count = 0

    elif front_active and rear_active:

        if find(0) == find(1):
            new_component_count = 1
        else:
            new_component_count = 2

    else:

        new_component_count = 1

    new_state = (
        new_front_status,
        new_rear_status,
        new_component_count,
    )

    return new_state, vertical_cost


# ============================================================
# HORIZONTAL TRANSITION
# ============================================================

def _horizontal_transition(
    state,
    front_edges,
    rear_edges,
):
    """
    Move the DP frontier from one aisle to the next.

    Returns
    -------
    tuple or None
        New state at the next aisle.

        None is returned when the transition would:

        - leave an old endpoint with odd degree, or
        - permanently close a disconnected component.
    """

    (
        front_status,
        rear_status,
        component_count,
    ) = state

    front_edge_status = _edge_status(
        front_edges
    )

    rear_edge_status = _edge_status(
        rear_edges
    )

    # --------------------------------------------------------
    # Final degrees of the OLD frontier vertices.
    # --------------------------------------------------------
    #
    # Once the frontier moves to the next aisle, these vertices
    # can never receive another edge. Their degrees must
    # therefore be even.

    old_front_final = _combine_degree_status(
        front_status,
        front_edge_status,
    )

    old_rear_final = _combine_degree_status(
        rear_status,
        rear_edge_status,
    )

    if (
        old_front_final == ODD
        or old_rear_final == ODD
    ):
        return None

    # --------------------------------------------------------
    # Small union-find:
    #
    #   0 = old front
    #   1 = old rear
    #   2 = new front
    #   3 = new rear
    # --------------------------------------------------------

    parent = list(
        range(4)
    )

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    old_front_active = (
        front_status != ZERO
    )

    old_rear_active = (
        rear_status != ZERO
    )

    # Existing connectivity of old frontier.
    if (
        component_count == 1
        and old_front_active
        and old_rear_active
    ):
        union(0, 1)

    # Horizontal front connection.
    if front_edges > 0:
        union(0, 2)

    # Horizontal rear connection.
    if rear_edges > 0:
        union(1, 3)

    # --------------------------------------------------------
    # An old vertex may have been zero before the horizontal
    # transition but become active because two horizontal edges
    # have just been attached to it.
    # --------------------------------------------------------

    old_front_final_active = (
        old_front_final != ZERO
    )

    old_rear_final_active = (
        old_rear_final != ZERO
    )

    new_front_active = (
        front_edges > 0
    )

    new_rear_active = (
        rear_edges > 0
    )

    # --------------------------------------------------------
    # Prevent a connected component from being "sealed off".
    #
    # Every active component containing an old frontier vertex
    # must also contain at least one NEW frontier vertex.
    #
    # Otherwise that component could never be connected to
    # anything processed later.
    # --------------------------------------------------------

    old_vertices = [
        (0, old_front_final_active),
        (1, old_rear_final_active),
    ]

    new_vertices = [
        (2, new_front_active),
        (3, new_rear_active),
    ]

    for old_node, old_is_active in old_vertices:

        if not old_is_active:
            continue

        root = find(
            old_node
        )

        survives = any(
            new_is_active
            and find(new_node) == root
            for new_node, new_is_active
            in new_vertices
        )

        if not survives:
            return None

    # --------------------------------------------------------
    # Degree states at NEW frontier.
    # --------------------------------------------------------

    new_front_status = _edge_status(
        front_edges
    )

    new_rear_status = _edge_status(
        rear_edges
    )

    if (
        not new_front_active
        and not new_rear_active
    ):

        new_component_count = 0

    elif (
        new_front_active
        and new_rear_active
    ):

        if find(2) == find(3):
            new_component_count = 1
        else:
            new_component_count = 2

    else:

        new_component_count = 1

    return (
        new_front_status,
        new_rear_status,
        new_component_count,
    )


# ============================================================
# EXACT RATLiff-ROSENTHAL DISTANCE
# ============================================================

def ratliff_rosenthal_distance(
    grid,
    depot,
    picks,
):
    """
    Compute the exact picker-route distance for a single-block,
    two-cross-aisle rectangular warehouse using an
    aisle-by-aisle Ratliff-Rosenthal-style dynamic program.

    Parameters
    ----------
    grid : np.ndarray
        Binary warehouse grid:
            0 = traversable
            1 = obstacle

    depot : tuple
        Depot coordinate as (row, column).

    picks : iterable
        Required pick coordinates.

    Returns
    -------
    float
        Exact minimum closed-tour distance.

    Notes
    -----
    The dynamic program maintains only information relevant to
    extending a partial route:

        - front endpoint degree state: 0 / E / U
        - rear endpoint degree state: 0 / E / U
        - number of frontier-connected components

    Dominated partial routes with the same state are discarded.
    """

    picks = list(
        picks
    )

    # --------------------------------------------------------
    # Extract warehouse geometry.
    # --------------------------------------------------------

    (
        front_row,
        rear_row,
        aisle_columns,
    ) = get_warehouse_geometry(
        grid
    )

    aisle_columns = sorted(
        aisle_columns
    )

    # --------------------------------------------------------
    # Validate depot.
    # --------------------------------------------------------

    if grid[depot] != 0:
        raise ValueError(
            "Depot must be located on a traversable cell."
        )

    if depot[0] != front_row:
        raise ValueError(
            "Depot must be located on the front cross aisle."
        )

    if depot[1] >= aisle_columns[0]:
        raise ValueError(
            "This implementation assumes the depot is located "
            "to the left of the picking aisles."
        )

    # --------------------------------------------------------
    # Empty order.
    # --------------------------------------------------------

    if not picks:
        return 0.0

    # --------------------------------------------------------
    # Validate pick locations.
    # --------------------------------------------------------

    if len(picks) != len(set(picks)):
        raise ValueError(
            "Order contains duplicate pick locations."
        )

    for pick in picks:

        row, col = pick

        if not (
            0 <= row < grid.shape[0]
            and 0 <= col < grid.shape[1]
        ):
            raise ValueError(
                f"Pick {pick} is outside the warehouse grid."
            )

        if grid[pick] != 0:
            raise ValueError(
                f"Pick {pick} is not on a traversable cell."
            )

        if col not in aisle_columns:
            raise ValueError(
                f"Pick {pick} is not in a valid picking aisle."
            )

        if not (
            front_row < row < rear_row
        ):
            raise ValueError(
                f"Pick {pick} is not inside the picking area."
            )

    # --------------------------------------------------------
    # Group picks by aisle.
    # --------------------------------------------------------

    grouped_picks = group_picks_by_aisle(
        picks,
        aisle_columns,
    )

    # Create an entry for every aisle because an EMPTY aisle
    # may still be useful as part of an optimal route.
    picks_by_aisle = {
        col: grouped_picks.get(col, [])
        for col in aisle_columns
    }

    # --------------------------------------------------------
    # Only process as far as the rightmost aisle containing a
    # required pick.
    # --------------------------------------------------------

    rightmost_required_aisle = max(
        col
        for _, col in picks
    )

    processing_aisles = [
        col
        for col in aisle_columns
        if col <= rightmost_required_aisle
    ]

    if not processing_aisles:
        raise RuntimeError(
            "No picking aisles available for the order."
        )

    # ========================================================
    # INITIAL CONNECTION FROM DEPOT
    # ========================================================
    #
    # The depot lies to the left of the warehouse.
    #
    # In a closed Eulerian tour, the cut between the depot and
    # warehouse must be crossed an even number of times.
    #
    # The non-dominated initial connection therefore consists
    # of two copies of the front cross-aisle path between the
    # depot and first picking aisle.

    first_aisle = processing_aisles[0]

    initial_horizontal_distance = (
        first_aisle - depot[1]
    )

    initial_state = (
        EVEN,
        ZERO,
        1,
    )

    dp = {
        initial_state:
            float(
                2 * initial_horizontal_distance
            )
    }

    # --------------------------------------------------------
    # Service first aisle.
    # --------------------------------------------------------

    first_dp = {}

    configs = aisle_service_configs(
        picks_by_aisle[first_aisle],
        front_row,
        rear_row,
    )

    for state, current_cost in dp.items():

        for config in configs:

            (
                new_state,
                vertical_cost,
            ) = _apply_vertical_config(
                state,
                config,
            )

            new_cost = (
                current_cost
                + vertical_cost
            )

            if (
                new_state not in first_dp
                or new_cost < first_dp[new_state]
            ):
                first_dp[new_state] = new_cost

    dp = first_dp

    previous_aisle = first_aisle

    # ========================================================
    # PROCESS REMAINING AISLES LEFT TO RIGHT
    # ========================================================

    for aisle_col in processing_aisles[1:]:

        horizontal_distance = (
            aisle_col
            - previous_aisle
        )

        # ----------------------------------------------------
        # Inter-aisle sub-stage
        # ----------------------------------------------------

        horizontal_dp = {}

        for state, current_cost in dp.items():

            for (
                front_edges,
                rear_edges,
            ) in HORIZONTAL_CONFIGS:

                new_state = _horizontal_transition(
                    state,
                    front_edges,
                    rear_edges,
                )

                if new_state is None:
                    continue

                added_horizontal_cost = (
                    front_edges
                    + rear_edges
                ) * horizontal_distance

                new_cost = (
                    current_cost
                    + added_horizontal_cost
                )

                if (
                    new_state not in horizontal_dp
                    or new_cost
                    < horizontal_dp[new_state]
                ):
                    horizontal_dp[new_state] = (
                        new_cost
                    )

        # ----------------------------------------------------
        # Intra-aisle sub-stage
        # ----------------------------------------------------

        vertical_dp = {}

        configs = aisle_service_configs(
            picks_by_aisle[aisle_col],
            front_row,
            rear_row,
        )

        for (
            state,
            current_cost,
        ) in horizontal_dp.items():

            for config in configs:

                (
                    new_state,
                    vertical_cost,
                ) = _apply_vertical_config(
                    state,
                    config,
                )

                new_cost = (
                    current_cost
                    + vertical_cost
                )

                if (
                    new_state not in vertical_dp
                    or new_cost
                    < vertical_dp[new_state]
                ):
                    vertical_dp[new_state] = (
                        new_cost
                    )

        dp = vertical_dp

        previous_aisle = aisle_col

    # ========================================================
    # FINAL VALID TOUR
    # ========================================================
    #
    # At the end:
    #
    # 1. no endpoint may have odd degree;
    # 2. all used edges must belong to one component.
    #
    # The depot component has been propagated through the
    # frontier throughout the dynamic program.

    best_distance = np.inf

    for (
        front_status,
        rear_status,
        component_count,
    ), cost in dp.items():

        if front_status == ODD:
            continue

        if rear_status == ODD:
            continue

        if component_count != 1:
            continue

        best_distance = min(
            best_distance,
            cost,
        )

    if np.isinf(best_distance):
        raise RuntimeError(
            "Ratliff-Rosenthal dynamic program found "
            "no valid complete route."
        )

    return float(
        best_distance
    )


# ============================================================
# OPTIONAL STANDARD INTERFACE
# ============================================================

def exact_optimal_distance(
    grid,
    depot,
    picks,
):
    """
    Alias for ratliff_rosenthal_distance().
    """

    return ratliff_rosenthal_distance(
        grid,
        depot,
        picks,
    )
