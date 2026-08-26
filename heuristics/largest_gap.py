import numpy as np


def largest_gap_route(grid, depot, picks):
    """
    Compute the total travel distance for the Largest Gap heuristic.

    Assumptions
    -----------
    - Single-block rectangular warehouse.
    - Front cross aisle is the first fully traversable row.
    - Rear cross aisle is the last fully traversable row.
    - Picking aisles are fully traversable vertical columns.
    - Depot is located on the front cross aisle.
    - All pick locations lie within valid picking aisles.

    Largest Gap routing
    -------------------
    - If there are no required picks, distance is 0.
    - If there is only one required aisle, Return routing is used.
    - If there are two or more required aisles:
        * the first required aisle is traversed completely;
        * the last required aisle is traversed completely;
        * intermediate required aisles are serviced from both ends,
          leaving the largest gap untraversed.

    Parameters
    ----------
    grid : np.ndarray
        2D binary array:
            0 = traversable
            1 = obstacle

    depot : tuple
        Depot coordinate as (row, column).

    picks : iterable
        Required pick coordinates as (row, column).

    Returns
    -------
    float
        Total travel distance in grid units.
    """

    H, W = grid.shape

    # --------------------------------------------------------
    # Identify vertical picking aisles
    # --------------------------------------------------------

    aisle_cols = [
        col
        for col in range(W)
        if np.all(grid[:, col] == 0)
    ]

    # --------------------------------------------------------
    # Identify front and rear cross aisles
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
    # Validate depot
    # --------------------------------------------------------

    if depot[0] != front_row:
        raise ValueError(
            "Largest Gap routing expects the depot "
            "to be located on the front cross aisle."
        )

    # --------------------------------------------------------
    # Group picks by aisle
    # --------------------------------------------------------

    picks_by_aisle = {}

    for pick in picks:

        row, col = pick

        if col not in aisle_cols:
            raise ValueError(
                f"Pick {pick} is not located "
                f"in a valid picking aisle."
            )

        if not (front_row < row < rear_row):
            raise ValueError(
                f"Pick {pick} is not inside the picking area."
            )

        picks_by_aisle.setdefault(
            col,
            []
        ).append(row)

    # --------------------------------------------------------
    # No picks
    # --------------------------------------------------------

    if not picks_by_aisle:
        return 0.0

    aisles_with_picks = sorted(
        picks_by_aisle.keys()
    )

    # --------------------------------------------------------
    # Single required aisle
    # --------------------------------------------------------
    # With only one required aisle, use Return routing:
    # enter from front, reach deepest pick, return to front.

    if len(aisles_with_picks) == 1:

        aisle_col = aisles_with_picks[0]

        deepest_row = max(
            picks_by_aisle[aisle_col]
        )

        horizontal_to_aisle = abs(
            depot[1] - aisle_col
        )

        vertical_distance = 2 * (
            deepest_row - front_row
        )

        horizontal_to_depot = abs(
            aisle_col - depot[1]
        )

        total_distance = (
            horizontal_to_aisle
            + vertical_distance
            + horizontal_to_depot
        )

        return float(total_distance)

    # --------------------------------------------------------
    # Two or more required aisles
    # --------------------------------------------------------

    first_aisle = aisles_with_picks[0]
    last_aisle = aisles_with_picks[-1]

    aisle_length = (
        rear_row - front_row
    )

    total_distance = 0.0

    # --------------------------------------------------------
    # Horizontal travel
    # --------------------------------------------------------
    #
    # Front:
    # depot -> first required aisle
    #
    # Rear:
    # first required aisle -> last required aisle
    #
    # Front:
    # last required aisle -> depot

    total_distance += abs(
        depot[1] - first_aisle
    )

    total_distance += abs(
        last_aisle - first_aisle
    )

    total_distance += abs(
        last_aisle - depot[1]
    )

    # --------------------------------------------------------
    # First and last required aisles
    # --------------------------------------------------------
    #
    # Both are traversed completely.

    total_distance += aisle_length
    total_distance += aisle_length

    # --------------------------------------------------------
    # Intermediate required aisles
    # --------------------------------------------------------

    for aisle_col in aisles_with_picks[1:-1]:

        rows = sorted(
            picks_by_aisle[aisle_col]
        )

        # Include both cross aisles so that gaps at
        # the front and rear ends are considered.
        points = (
            [front_row]
            + rows
            + [rear_row]
        )

        # Calculate gaps between every consecutive pair.
        gaps = [
            points[i + 1] - points[i]
            for i in range(len(points) - 1)
        ]

        largest_gap = max(gaps)

        # The largest gap is not traversed.
        #
        # Everything else is approached from the
        # front or rear and returned along the same side.
        vertical_distance = 2 * (
            aisle_length - largest_gap
        )

        total_distance += vertical_distance

    return float(total_distance)
