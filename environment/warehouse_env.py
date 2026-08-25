import numpy as np


class WarehouseEnv:
    def __init__(
        self,
        grid,
        depot,
        all_pick_locations,
        max_steps=500,
        move_cost=-1.0,
        invalid_penalty=-2.0,
        pick_reward=2.0,
        completion_reward=20.0
    ):
        self.grid = grid
        self.depot = depot

        self.all_pick_locations = tuple(sorted(set(all_pick_locations)))
        self.all_pick_location_set = set(self.all_pick_locations)

        self.pick_index = {
            location: index
            for index, location in enumerate(self.all_pick_locations)
        }

        self.max_steps = max_steps

        self.move_cost = move_cost
        self.invalid_penalty = invalid_penalty
        self.pick_reward = pick_reward
        self.completion_reward = completion_reward

        self.agent_pos = self.depot
        self.current_order = set()
        self.remaining_picks = set()
        self.collected_picks = set()

        self.steps = 0
        self.travel_distance = 0

        self.terminated = False
        self.truncated = False
        self.done = False

    def reset(self, order):
        """
        Start a new picking episode.

        Parameters
        ----------
        order : iterable
            Collection of unique pick-location coordinates.
        """

        order = list(order)

        if len(order) != len(set(order)):
            raise ValueError("Order contains duplicate pick locations.")

        unknown_locations = set(order) - self.all_pick_location_set

        if unknown_locations:
            raise ValueError(
                f"Order contains invalid pick locations: {unknown_locations}"
            )

        self.agent_pos = self.depot

        self.current_order = set(order)
        self.remaining_picks = set(order)
        self.collected_picks = set()

        self.steps = 0
        self.travel_distance = 0

        self.terminated = False
        self.truncated = False
        self.done = False

        return self._get_state()

    def step(self, action):
        """
        Take one movement action.

        Actions
        -------
        0 = up
        1 = down
        2 = left
        3 = right
        """

        if self.done:
            raise ValueError(
                "Episode already finished. Call reset() before taking another step."
            )

        if action not in [0, 1, 2, 3]:
            raise ValueError(
                "Action must be one of: 0=up, 1=down, 2=left, 3=right"
            )

        movements = [
            (-1, 0),   # up
            (1, 0),    # down
            (0, -1),   # left
            (0, 1)
        ]

        dr, dc = movements[action]

        new_pos = (
            self.agent_pos[0] + dr,
            self.agent_pos[1] + dc
        )

        moved = False

        if self._is_valid(new_pos):
            self.agent_pos = new_pos

            self.travel_distance += 1

            reward = self.move_cost
            moved = True

        else:
            reward = self.invalid_penalty

        self.steps += 1

        if self.agent_pos in self.remaining_picks:
            self.remaining_picks.remove(self.agent_pos)
            self.collected_picks.add(self.agent_pos)

            reward += self.pick_reward

        if (
            len(self.remaining_picks) == 0
            and self.agent_pos == self.depot
        ):
            self.terminated = True
            reward += self.completion_reward

        if self.steps >= self.max_steps and not self.terminated:
            self.truncated = True

        self.done = self.terminated or self.truncated

        info = {
            "travel_distance": self.travel_distance,
            "steps": self.steps,
            "moved": moved,
            "remaining_picks": len(self.remaining_picks),
            "collected_picks": len(self.collected_picks)
        }

        return (
            self._get_state(),
            reward,
            self.terminated,
            self.truncated,
            info
        )

    def _is_valid(self, pos):
        """
        Check whether a coordinate is a valid walkable grid cell.
        """

        r, c = pos

        if not (
            0 <= r < self.grid.shape[0]
            and 0 <= c < self.grid.shape[1]
        ):
            return False

        if self.grid[r, c] == 1:
            return False

        return True

    def _get_state(self):
        """
        Construct the DQN state:

        [normalised_row,
         normalised_column,
         pick_1,
         pick_2,
         ...
         pick_96]

        pick_i:
            1 = still required
            0 = not required / already collected
        """

        H, W = self.grid.shape

        state = np.zeros(
            2 + len(self.all_pick_locations),
            dtype=np.float32
        )

        row, col = self.agent_pos

        state[0] = row / (H - 1)
        state[1] = col / (W - 1)

        for location in self.remaining_picks:
            index = self.pick_index[location]

            state[2 + index] = 1.0

        return state
