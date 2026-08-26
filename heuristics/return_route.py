import numpy as np


def return_route(grid, depot, picks):
    """
    Compute the total travel distance for the Return routing heuristic.

    Assumptions:
    - The warehouse is a single-block rectangular warehouse.
    - The front cross aisle is the first fully traversable row.
    - The rear cross aisle is the last fully traversable row.
    - Picking aisles are vertical fully traversable columns.
    - The depot is located on the front cross aisle.
    - Pick locations lie inside valid picking aisles.

    Return routing:
    - Move from the depot along the front cross aisle.
    - Enter each aisle containing picks.
    - Travel to the deepest required pick in that aisle.
    - Return through the same aisle to the front cross aisle.
    - Continue to the next required aisle.
    - Return to the depot after the final aisle.

    Parameters
    ----------
    grid : np.ndarray
        2D array where:
        0 = walkable
        1 = obstacle

    depot : tuple
        Depot coordinate as (row, column).

    picks : iterable
        Required pick coordinates.

    Returns
    -------
    float
        Total travel distance.
    """

    H, W = grid.shape

    # --------------------------------------------------------
    # Find picking aisles
    # --------------------------------------------------------

    aisle_cols = [
        col
        for col in range(W)
        if np.all(grid[:, col] == 0)
    ]

    # --------------------------------------------------------
    # Find cross aisles
    # --------------------------------------------------------

    cross_rows = [
        row
        for row in range(H)
        if np.all(grid[row, :] == 0)
    ]

    if len(cross_rows) < 2:
        raise ValueError(
            "Warehouse must contain a front and rear cross aisle."
        )

    front_row = min(cross_rows)
    rear_row = max(cross_rows)

    # --------------------------------------------------------
    # Check depot
    # --------------------------------------------------------

    if depot[0] != front_row:
        raise ValueError(
            "Depot must be located on the front cross aisle."
        )

    # --------------------------------------------------------
    # Group picks by aisle
    # --------------------------------------------------------

    picks_by_aisle = {}

    for pick in picks:

        row, col = pick

        if col not in aisle_cols:
            raise ValueError(
                f"Pick {pick} is not located in a valid picking aisle."
            )

        if not (front_row < row < rear_row):
            raise ValueError(
                f"Pick {pick} is not inside the picking area."
            )

        picks_by_aisle.setdefault(
            col,
            []
        ).append(pick)

    # --------------------------------------------------------
    # No picks
    # --------------------------------------------------------

    if not picks_by_aisle:
        return 0.0

    # Visit required aisles from left to right.
    aisles_with_picks = sorted(
        picks_by_aisle.keys()
    )

    total_distance = 0.0

    # Picker starts at depot.
    current_col = depot[1]

    # --------------------------------------------------------
    # Visit required aisles
    # --------------------------------------------------------

    for aisle_col in aisles_with_picks:

        # Horizontal movement along front cross aisle
        total_distance += abs(
            current_col - aisle_col
        )

        # Deepest required pick in this aisle
        deepest_row = max(
            pick[0]
            for pick in picks_by_aisle[aisle_col]
        )

        # Distance from front cross aisle to deepest pick
        one_way_distance = (
            deepest_row - front_row
        )

        # Travel into aisle and return through same aisle
        total_distance += (
            2 * one_way_distance
        )

        # Back on front cross aisle at this aisle column
        current_col = aisle_col

    # --------------------------------------------------------
    # Return to depot
    # --------------------------------------------------------

    total_distance += abs(
        current_col - depot[1]
    )

    return float(total_distance)
