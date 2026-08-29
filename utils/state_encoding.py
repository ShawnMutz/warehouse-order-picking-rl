"""
Coordinate-based DQN state encoding.

The WarehouseEnv retains its original state representation:

    [normalised_picker_row,
     normalised_picker_col,
     remaining_pick_bit_1,
     ...
     remaining_pick_bit_96]

This module converts that environment state into a more explicit
spatial representation for the DQN.

For a maximum order size of 20:

    2 picker coordinates
    +
    20 pick slots x 3 values

Each pick slot contains:

    normalised_pick_row
    normalised_pick_col
    active_flag

Total:

    2 + (20 * 3) = 62 dimensions

Important
---------
A CoordinateStateEncoder is configured with the original order at
the beginning of an episode using set_order().

Each original pick is then assigned a fixed slot for that episode.

When a pick is collected:
    - its slot becomes [0, 0, 0]
    - other remaining picks DO NOT shift between slots

This gives a stable coordinate-based representation throughout
the episode.
"""

import numpy as np


# ============================================================
# COORDINATE STATE ENCODER
# ============================================================

class CoordinateStateEncoder:
    """
    Convert the WarehouseEnv binary-location state into a
    fixed-size coordinate representation.
    """

    def __init__(
        self,
        all_pick_locations,
        grid_shape,
        max_order_size=20,
    ):
        """
        Parameters
        ----------
        all_pick_locations : iterable of tuple[int, int]
            The fixed warehouse pick-location list used by
            WarehouseEnv.

        grid_shape : tuple[int, int]
            Warehouse grid shape:
                (rows, columns)

        max_order_size : int
            Maximum number of picks that can be represented.

            For the final experiment:
                max_order_size = 20
        """

        self.all_pick_locations = list(
            all_pick_locations
        )

        self.grid_shape = tuple(
            grid_shape
        )

        self.max_order_size = int(
            max_order_size
        )

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if len(
            self.grid_shape
        ) != 2:
            raise ValueError(
                "grid_shape must contain exactly two values: "
                "(rows, columns)."
            )

        self.rows = int(
            self.grid_shape[0]
        )

        self.cols = int(
            self.grid_shape[1]
        )

        if self.rows <= 1:
            raise ValueError(
                "Warehouse must contain at least two rows."
            )

        if self.cols <= 1:
            raise ValueError(
                "Warehouse must contain at least two columns."
            )

        if self.max_order_size <= 0:
            raise ValueError(
                "max_order_size must be greater than zero."
            )

        if len(
            self.all_pick_locations
        ) == 0:
            raise ValueError(
                "all_pick_locations cannot be empty."
            )

        # ----------------------------------------------------
        # Stable map:
        #
        # warehouse location -> binary state index
        # ----------------------------------------------------

        # ----------------------------------------------------
        # WarehouseEnv constructs its binary state vector using
        # locations in sorted coordinate order.
        #
        # The original all_pick_locations list is column-major:
        #
        #     (1,1), (2,1), ..., (12,1),
        #     (1,3), (2,3), ...
        #
        # while the environment state uses row-major sorted order:
        #
        #     (1,1), (1,3), ..., (1,15),
        #     (2,1), (2,3), ...
        #
        # The encoder must therefore use the same ordering when
        # mapping a location to its state-vector bit.
        # ----------------------------------------------------

        self.environment_state_locations = sorted(
            tuple(location)
            for location in self.all_pick_locations
        )

        self.location_to_index = {
            location: index
            for index, location
            in enumerate(
                self.environment_state_locations
            )
        }

        if len(
            self.location_to_index
        ) != len(
            self.all_pick_locations
        ):
            raise ValueError(
                "all_pick_locations contains duplicate "
                "locations."
            )

        # ----------------------------------------------------
        # Input/output dimensions
        # ----------------------------------------------------

        self.environment_state_dim = (
            2
            + len(
                self.all_pick_locations
            )
        )

        self.output_dim = (
            2
            + 3
            * self.max_order_size
        )

        # ----------------------------------------------------
        # Episode order
        # ----------------------------------------------------

        self.order = None

    # ========================================================
    # EPISODE ORDER
    # ========================================================

    def set_order(
        self,
        order,
    ):
        """
        Configure the encoder for one episode.

        The order is sorted into a deterministic coordinate
        order so that supplying the same set of picks in a
        different Python-list order gives the same encoding.

        Each pick retains its assigned slot throughout the
        episode.
        """

        order = [
            tuple(
                location
            )
            for location in order
        ]

        # ----------------------------------------------------
        # Maximum order size
        # ----------------------------------------------------

        if len(order) > self.max_order_size:

            raise ValueError(
                f"Order contains {len(order)} picks, but "
                f"max_order_size is "
                f"{self.max_order_size}."
            )

        # ----------------------------------------------------
        # Duplicate picks
        # ----------------------------------------------------

        if len(
            set(
                order
            )
        ) != len(
            order
        ):

            raise ValueError(
                "Order contains duplicate pick locations."
            )

        # ----------------------------------------------------
        # Valid warehouse locations
        # ----------------------------------------------------

        invalid_locations = [
            location
            for location in order
            if location
            not in self.location_to_index
        ]

        if invalid_locations:

            raise ValueError(
                "Order contains locations that are not "
                "registered warehouse pick locations: "
                f"{invalid_locations}"
            )

        # ----------------------------------------------------
        # Stable slot assignment
        # ----------------------------------------------------

        self.order = sorted(
            order
        )

    # ========================================================
    # RESET ALIAS
    # ========================================================

    def reset(
        self,
        order,
    ):
        """
        Alias for set_order().

        This makes training-loop usage intuitive:

            env_state = env.reset(order)
            encoder.reset(order)
            dqn_state = encoder.encode(env_state)
        """

        self.set_order(
            order
        )

    # ========================================================
    # NORMALISATION
    # ========================================================

    def normalise_location(
        self,
        location,
    ):
        """
        Convert a discrete warehouse coordinate into
        normalised row/column values in [0, 1].
        """

        row, col = location

        normalised_row = (
            float(row)
            / float(
                self.rows - 1
            )
        )

        normalised_col = (
            float(col)
            / float(
                self.cols - 1
            )
        )

        return (
            normalised_row,
            normalised_col,
        )

    # ========================================================
    # STATE VALIDATION
    # ========================================================

    def _validate_environment_state(
        self,
        state,
    ):
        """
        Validate and convert one WarehouseEnv state.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.ndim != 1:

            raise ValueError(
                "Environment state must be one-dimensional."
            )

        if len(
            state
        ) != self.environment_state_dim:

            raise ValueError(
                "Environment state has incorrect dimension: "
                f"expected {self.environment_state_dim}, "
                f"received {len(state)}."
            )

        if not np.all(
            np.isfinite(
                state
            )
        ):

            raise ValueError(
                "Environment state contains non-finite values."
            )

        return state

    # ========================================================
    # CHECK WHETHER A PICK REMAINS
    # ========================================================

    def is_pick_active(
        self,
        environment_state,
        location,
    ):
        """
        Return True if the supplied pick location remains
        uncollected in the WarehouseEnv state.
        """

        location = tuple(
            location
        )

        location_index = (
            self.location_to_index[
                location
            ]
        )

        binary_state_index = (
            2
            + location_index
        )

        return bool(
            environment_state[
                binary_state_index
            ]
            > 0.5
        )

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        environment_state,
    ):
        """
        Convert one WarehouseEnv state into the coordinate
        representation.

        Output layout
        -------------

        index 0:
            picker normalised row

        index 1:
            picker normalised column

        then for each of max_order_size slots:

            normalised pick row
            normalised pick column
            active flag

        Inactive and unused slots are:

            [0.0, 0.0, 0.0]

        Returns
        -------
        np.ndarray
            float32 state vector of fixed length output_dim.
        """

        if self.order is None:

            raise RuntimeError(
                "Encoder has no episode order. Call "
                "set_order(order) or reset(order) before "
                "encode()."
            )

        environment_state = (
            self._validate_environment_state(
                environment_state
            )
        )

        encoded_state = np.zeros(
            self.output_dim,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Picker coordinates
        #
        # WarehouseEnv already supplies these normalised.
        # ----------------------------------------------------

        encoded_state[0] = (
            environment_state[0]
        )

        encoded_state[1] = (
            environment_state[1]
        )

        # ----------------------------------------------------
        # Fixed pick slots
        # ----------------------------------------------------

        for (
            slot_index,
            location,
        ) in enumerate(
            self.order
        ):

            active = self.is_pick_active(
                environment_state,
                location,
            )

            # ------------------------------------------------
            # If already collected, leave slot as:
            #
            # [0, 0, 0]
            # ------------------------------------------------

            if not active:
                continue

            (
                normalised_row,
                normalised_col,
            ) = self.normalise_location(
                location
            )

            slot_start = (
                2
                + 3
                * slot_index
            )

            encoded_state[
                slot_start
            ] = normalised_row

            encoded_state[
                slot_start + 1
            ] = normalised_col

            encoded_state[
                slot_start + 2
            ] = 1.0

        return encoded_state

    # ========================================================
    # CONVENIENCE
    # ========================================================

    def get_output_dim(self):
        """
        Return the coordinate-state dimensionality.
        """

        return self.output_dim
