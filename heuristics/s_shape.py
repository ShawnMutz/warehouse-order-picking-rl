import numpy as np


def s_shape_route(grid, depot, picks):
    """
    Calculate the total travel distance using the S-shape heuristic.

    Assumptions:
    - Front cross aisle is the first fully walkable row.
    - Rear cross aisle is the last fully walkable row.
    - Picking aisles are fully walkable vertical columns.
    - Depot is on the front cross aisle.
    - Every aisle containing at least one pick is fully traversed.
    - Successive aisles are traversed in alternating directions.
    """

    H, W = grid.shape

    # Find vertical picking aisles
    aisle_cols = [
        col
        for col in range(W)
        if np.all(grid[:, col] == 0)
    ]

    # Find front and rear cross aisles
    cross_rows = [
        row
        for row in range(H)
        if np.all(grid[row, :] == 0)
    ]

    if len(cross_rows) < 2:
        raise ValueError(
            "Warehouse must have a front and rear cross aisle."
        )

    front_row = min(cross_rows)
    rear_row = max(cross_rows)

    # Depot should be on front cross aisle
    if depot[0] != front_row:
        raise ValueError(
            "Depot must be located on the front cross aisle."
        )

    # Group picks by aisle
    picks_by_aisle = {}

    for pick in picks:
        row, col = pick

        if col not in aisle_cols:
            raise ValueError(
                f"Pick {pick} is not located in a valid picking aisle."
            )

        picks_by_aisle.setdefault(col, []).append(pick)

    # Required aisles from left to right
    aisles_with_picks = sorted(picks_by_aisle.keys())

    # No order = no travel
    if not aisles_with_picks:
        return 0.0

    total_distance = 0.0

    current_row, current_col = depot

    # First required aisle is entered from front
    entering_from_front = True

    aisle_length = rear_row - front_row

    for aisle_col in aisles_with_picks:

        # Move horizontally along current cross aisle
        total_distance += abs(
            current_col - aisle_col
        )

        # Fully traverse required aisle
        total_distance += aisle_length

        # Update picker location
        if entering_from_front:
            current_row = rear_row
        else:
            current_row = front_row

        current_col = aisle_col

        # Alternate direction
        entering_from_front = not entering_from_front

    # Return to depot
    if current_row == front_row:

        # Already at front cross aisle
        total_distance += abs(
            current_col - depot[1]
        )

    else:

        # At rear cross aisle:
        # return through final aisle to front
        total_distance += aisle_length

        # Then travel along front cross aisle to depot
        total_distance += abs(
            current_col - depot[1]
        )

    return float(total_distance)
