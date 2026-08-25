import numpy as np


def generate_order(
    all_pick_locations,
    size,
    distribution,
    seed=None,
    aisle_columns=None
):
    """
    Generate a warehouse picking order.

    Parameters
    ----------
    all_pick_locations : iterable
        All fixed warehouse pick locations.
        For the final warehouse there should be 96 locations.

    size : int
        Number of unique picks required in the order.
        Expected values: 5, 10, 15 or 20.

    distribution : str
        Order-generation distribution.

        "uniform"
            Picks are sampled uniformly from all 96 locations.

        "clustered"
            Picks are sampled from a randomly selected group
            of three contiguous picking aisles.

    seed : int, optional
        Random seed used to make order generation reproducible.

    aisle_columns : list, optional
        Column coordinates of the eight picking aisles.

        Default:
        [1, 3, 5, 7, 9, 11, 13, 15]

    Returns
    -------
    list of tuple
        List of unique (row, column) pick locations.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------
    # Fixed warehouse aisle locations
    # ---------------------------------
    if aisle_columns is None:
        aisle_columns = [1, 3, 5, 7, 9, 11, 13, 15]

    # Sort locations so that seeded generation is reproducible
    # even if a set was supplied.
    all_pick_locations = sorted(set(all_pick_locations))

    # ---------------------------------
    # Validate inputs
    # ---------------------------------
    if size <= 0:
        raise ValueError("Order size must be greater than zero.")

    if size > len(all_pick_locations):
        raise ValueError(
            f"Cannot generate an order of size {size}. "
            f"Only {len(all_pick_locations)} pick locations exist."
        )

    if distribution not in ["uniform", "clustered"]:
        raise ValueError(
            "Distribution must be either 'uniform' or 'clustered'."
        )

    # =================================
    # UNIFORM DISTRIBUTION
    # =================================
    if distribution == "uniform":

        # Every one of the 96 storage locations has the
        # same probability of being selected.
        indices = rng.choice(
            len(all_pick_locations),
            size=size,
            replace=False
        )

        picks = [
            all_pick_locations[index]
            for index in indices
        ]

        return picks

    # =================================
    # CLUSTERED DISTRIBUTION
    # =================================
    if distribution == "clustered":

        cluster_width = 3

        if len(aisle_columns) < cluster_width:
            raise ValueError(
                "There are not enough aisles to create "
                "a three-aisle cluster."
            )

        # With 8 aisles and width 3, possible starting
        # positions are:
        #
        # 0 -> aisles 1,2,3
        # 1 -> aisles 2,3,4
        # 2 -> aisles 3,4,5
        # 3 -> aisles 4,5,6
        # 4 -> aisles 5,6,7
        # 5 -> aisles 6,7,8

        number_of_clusters = (
            len(aisle_columns) - cluster_width + 1
        )

        start_index = rng.integers(number_of_clusters)

        selected_aisles = aisle_columns[
            start_index:start_index + cluster_width
        ]

        # Keep only storage locations belonging
        # to the selected three aisles.
        candidates = [
            location
            for location in all_pick_locations
            if location[1] in selected_aisles
        ]

        if size > len(candidates):
            raise ValueError(
                f"Cannot generate clustered order of size {size}. "
                f"The selected cluster contains only "
                f"{len(candidates)} locations."
            )

        indices = rng.choice(
            len(candidates),
            size=size,
            replace=False
        )

        picks = [
            candidates[index]
            for index in indices
        ]

        return picks
