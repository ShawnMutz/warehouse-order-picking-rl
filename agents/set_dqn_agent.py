"""
Permutation-invariant Set-DQN agent for warehouse routing.

Motivation
----------
The earlier coordinate-state representation used 20 fixed pick slots:

    picker coordinates
    +
    slot 1
    slot 2
    ...
    slot 20

This worked strongly for one- and two-pick orders, but did not
generalise to unseen three-pick combinations.

A fixed-slot MLP is not naturally permutation invariant. In
particular, slots that are always zero during smaller-order
training become previously unseen inputs when larger orders are
introduced.

This module instead treats the remaining picks as a SET.

For every warehouse pick location, the network constructs:

    absolute normalised row
    absolute normalised column
    row relative to picker
    column relative to picker
    active flag

The same neural network is applied independently to every active
pick.

The resulting pick embeddings are mean-pooled, giving a fixed-size,
permutation-invariant representation of the remaining pick set.

The pooled representation is combined with:

    picker row
    picker column
    remaining-pick fraction

and passed through a DQN head producing four movement Q-values.

Architecture
------------
Per-pick encoder:

    5 -> 64 -> 64

Set aggregation:

    masked mean pooling

Global representation:

    picker row
    picker column
    remaining count / max order size
    64-dimensional pooled pick embedding

    = 67 features

DQN head:

    67 -> 256 -> 256 -> 4

The environment itself is unchanged.

The agent consumes the original raw WarehouseEnv state:

    2 picker coordinates
    +
    96 remaining-pick indicators
    =
    98 dimensions

Termination semantics
---------------------
Episode rollout stops on:

    terminated OR truncated

Bellman bootstrapping is masked ONLY for genuine termination.

A max-step truncation therefore ends the rollout but does not
incorrectly make the transition terminal.
"""

import random

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# REPLAY BUFFER
# ============================================================

class SetReplayBuffer:
    """
    Replay buffer storing raw WarehouseEnv states.

    Stored transition:

        state
        action
        reward
        next_state
        terminated

    Truncation is deliberately not stored as terminal because
    max-step truncation should not mask Bellman bootstrapping.
    """

    def __init__(
        self,
        capacity,
    ):
        capacity = int(
            capacity
        )

        if capacity <= 0:
            raise ValueError(
                "Replay-buffer capacity must be greater than zero."
            )

        self.capacity = capacity

        self.memory = deque(
            maxlen=capacity
        )

    def __len__(
        self,
    ):
        return len(
            self.memory
        )

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):
        """
        Store one transition.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        ).copy()

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        ).copy()

        self.memory.append(
            (
                state,
                int(
                    action
                ),
                float(
                    reward
                ),
                next_state,
                bool(
                    terminated
                ),
            )
        )

    def sample(
        self,
        batch_size,
    ):
        """
        Randomly sample a transition batch.
        """

        batch_size = int(
            batch_size
        )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if batch_size > len(
            self.memory
        ):
            raise ValueError(
                "Cannot sample more transitions than are "
                "currently stored."
            )

        batch = random.sample(
            self.memory,
            batch_size,
        )

        (
            states,
            actions,
            rewards,
            next_states,
            terminateds,
        ) = zip(
            *batch
        )

        return (
            np.stack(
                states
            ).astype(
                np.float32
            ),

            np.asarray(
                actions,
                dtype=np.int64,
            ),

            np.asarray(
                rewards,
                dtype=np.float32,
            ),

            np.stack(
                next_states
            ).astype(
                np.float32
            ),

            np.asarray(
                terminateds,
                dtype=np.float32,
            ),
        )


# ============================================================
# SET DQN NETWORK
# ============================================================

class SetDQN(nn.Module):
    """
    Permutation-invariant DQN network.

    Parameters
    ----------
    all_pick_locations
        All fixed warehouse pick coordinates.

    grid_shape
        Warehouse grid shape:

            (rows, columns)

    n_actions
        Number of movement actions.

    max_order_size
        Maximum supported order size.

    pick_embedding_dim
        Dimension of the shared per-pick embedding.

    hidden_dim
        Width of the DQN head.
    """

    def __init__(
        self,
        all_pick_locations,
        grid_shape,
        n_actions=4,
        max_order_size=20,
        pick_embedding_dim=64,
        hidden_dim=256,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Validate warehouse geometry
        # ----------------------------------------------------

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
                "Warehouse must contain at least two rows "
                "and two columns."
            )

        # ----------------------------------------------------
        # Validate locations
        # ----------------------------------------------------

        locations = [
            tuple(
                location
            )
            for location in all_pick_locations
        ]

        if not locations:
            raise ValueError(
                "all_pick_locations cannot be empty."
            )

        if len(
            locations
        ) != len(
            set(
                locations
            )
        ):
            raise ValueError(
                "all_pick_locations contains duplicates."
            )

        # WarehouseEnv stores its remaining-pick bits in
        # sorted coordinate order.
        self.environment_state_locations = sorted(
            locations
        )

        self.location_count = len(
            self.environment_state_locations
        )

        self.raw_state_dim = (
            2
            + self.location_count
        )

        # ----------------------------------------------------
        # Other settings
        # ----------------------------------------------------

        self.n_actions = int(
            n_actions
        )

        self.max_order_size = int(
            max_order_size
        )

        self.pick_embedding_dim = int(
            pick_embedding_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        if self.n_actions <= 0:
            raise ValueError(
                "n_actions must be greater than zero."
            )

        if self.max_order_size <= 0:
            raise ValueError(
                "max_order_size must be greater than zero."
            )

        if self.pick_embedding_dim <= 0:
            raise ValueError(
                "pick_embedding_dim must be greater than zero."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        # ----------------------------------------------------
        # Normalised fixed location coordinates
        # ----------------------------------------------------

        normalised_locations = []

        for row, col in self.environment_state_locations:

            normalised_locations.append(
                (
                    float(
                        row
                    )
                    / float(
                        rows - 1
                    ),

                    float(
                        col
                    )
                    / float(
                        cols - 1
                    ),
                )
            )

        location_coordinates = torch.tensor(
            normalised_locations,
            dtype=torch.float32,
        )

        # A registered buffer moves with the model when
        # .to(device) is called, but is not trainable.
        self.register_buffer(
            "location_coordinates",
            location_coordinates,
        )

        # ----------------------------------------------------
        # Shared pick encoder
        #
        # Input:
        #   absolute row
        #   absolute col
        #   relative row
        #   relative col
        #   active flag
        # ----------------------------------------------------

        self.pick_encoder = nn.Sequential(
            nn.Linear(
                5,
                self.pick_embedding_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.pick_embedding_dim,
                self.pick_embedding_dim,
            ),
            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Global DQN head
        #
        # picker row
        # picker col
        # remaining fraction
        # pooled pick embedding
        # ----------------------------------------------------

        global_input_dim = (
            3
            + self.pick_embedding_dim
        )

        self.q_head = nn.Sequential(
            nn.Linear(
                global_input_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim,
                self.n_actions,
            ),
        )

    # ========================================================
    # RAW STATE -> SET FEATURES
    # ========================================================

    def build_set_inputs(
        self,
        states,
    ):
        """
        Convert raw WarehouseEnv states into set-network inputs.

        Parameters
        ----------
        states : torch.Tensor
            Shape:

                [98]

            or:

                [batch, 98]

        Returns
        -------
        picker_coordinates
            [batch, 2]

        pick_features
            [batch, 96, 5]

        active_mask
            [batch, 96]
        """

        if not torch.is_tensor(
            states
        ):
            states = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=self.location_coordinates.device,
            )

        states = states.to(
            dtype=torch.float32,
            device=self.location_coordinates.device,
        )

        if states.ndim == 1:

            states = states.unsqueeze(
                0
            )

        if states.ndim != 2:

            raise ValueError(
                "states must have shape [state_dim] or "
                "[batch, state_dim]."
            )

        if states.shape[1] != self.raw_state_dim:

            raise ValueError(
                "Incorrect raw WarehouseEnv state dimension: "
                f"expected {self.raw_state_dim}, "
                f"received {states.shape[1]}."
            )

        if not torch.isfinite(
            states
        ).all():

            raise ValueError(
                "states contains non-finite values."
            )

        # ----------------------------------------------------
        # Picker coordinates
        # ----------------------------------------------------

        picker_coordinates = (
            states[
                :,
                0:2,
            ]
        )

        # ----------------------------------------------------
        # Remaining-pick bits
        # ----------------------------------------------------

        active_mask = (
            states[
                :,
                2:
            ]
            > 0.5
        ).to(
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Fixed absolute pick coordinates
        # ----------------------------------------------------

        batch_size = states.shape[
            0
        ]

        absolute_coordinates = (
            self.location_coordinates
            .unsqueeze(
                0
            )
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # ----------------------------------------------------
        # Coordinates relative to current picker position
        # ----------------------------------------------------

        relative_coordinates = (
            absolute_coordinates
            - picker_coordinates.unsqueeze(
                1
            )
        )

        # ----------------------------------------------------
        # Active feature
        # ----------------------------------------------------

        active_feature = (
            active_mask.unsqueeze(
                -1
            )
        )

        # ----------------------------------------------------
        # Per-location feature matrix
        # ----------------------------------------------------

        pick_features = torch.cat(
            (
                absolute_coordinates,
                relative_coordinates,
                active_feature,
            ),
            dim=-1,
        )

        return (
            picker_coordinates,
            pick_features,
            active_mask,
        )

    # ========================================================
    # SET FORWARD
    # ========================================================

    def forward_from_set(
        self,
        picker_coordinates,
        pick_features,
        active_mask,
    ):
        """
        Calculate Q-values from already constructed set inputs.

        This method is public so permutation invariance can be
        tested directly.

        Reordering the pick dimension while applying the same
        reordering to active_mask must leave the Q-values
        unchanged.
        """

        if picker_coordinates.ndim != 2:
            raise ValueError(
                "picker_coordinates must have shape [batch, 2]."
            )

        if picker_coordinates.shape[
            1
        ] != 2:
            raise ValueError(
                "picker_coordinates must contain two features."
            )

        if pick_features.ndim != 3:

            raise ValueError(
                "pick_features must have shape "
                "[batch, locations, features]."
            )

        if pick_features.shape[
            2
        ] != 5:

            raise ValueError(
                "Each pick must contain five features."
            )

        if active_mask.ndim != 2:

            raise ValueError(
                "active_mask must have shape "
                "[batch, locations]."
            )

        if (
            picker_coordinates.shape[
                0
            ]
            != pick_features.shape[
                0
            ]
            or
            picker_coordinates.shape[
                0
            ]
            != active_mask.shape[
                0
            ]
        ):

            raise ValueError(
                "Batch dimensions do not match."
            )

        if (
            pick_features.shape[
                1
            ]
            != active_mask.shape[
                1
            ]
        ):

            raise ValueError(
                "Pick and mask location dimensions do not match."
            )

        # ----------------------------------------------------
        # Apply the SAME encoder to every pick.
        # ----------------------------------------------------

        pick_embeddings = (
            self.pick_encoder(
                pick_features
            )
        )

        # ----------------------------------------------------
        # Mask inactive warehouse locations.
        # ----------------------------------------------------

        mask = active_mask.unsqueeze(
            -1
        )

        masked_embeddings = (
            pick_embeddings
            * mask
        )

        # ----------------------------------------------------
        # Permutation-invariant mean pooling.
        # ----------------------------------------------------

        remaining_count = (
            active_mask.sum(
                dim=1,
                keepdim=True,
            )
        )

        denominator = torch.clamp(
            remaining_count,
            min=1.0,
        )

        pooled_embedding = (
            masked_embeddings.sum(
                dim=1
            )
            / denominator
        )

        # If there are no remaining picks, the masked sum is
        # already exactly zero.
        remaining_fraction = (
            remaining_count
            / float(
                self.max_order_size
            )
        )

        # ----------------------------------------------------
        # Global routing representation
        # ----------------------------------------------------

        global_features = torch.cat(
            (
                picker_coordinates,
                remaining_fraction,
                pooled_embedding,
            ),
            dim=1,
        )

        return self.q_head(
            global_features
        )

    # ========================================================
    # STANDARD FORWARD
    # ========================================================

    def forward(
        self,
        states,
    ):
        """
        Calculate Q-values directly from raw WarehouseEnv states.
        """

        original_was_single = (
            torch.is_tensor(
                states
            )
            and states.ndim == 1
        )

        if not torch.is_tensor(
            states
        ):

            array = np.asarray(
                states
            )

            original_was_single = (
                array.ndim == 1
            )

        (
            picker_coordinates,
            pick_features,
            active_mask,
        ) = self.build_set_inputs(
            states
        )

        q_values = self.forward_from_set(
            picker_coordinates,
            pick_features,
            active_mask,
        )

        if original_was_single:

            return q_values.squeeze(
                0
            )

        return q_values


# ============================================================
# SET-DQN AGENT
# ============================================================

class SetDQNAgent:
    """
    DQN agent using the permutation-invariant SetDQN network.
    """

    def __init__(
        self,
        env,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99999,
        batch_size=64,
        buffer_capacity=50_000,
        target_update=1_000,
        hidden_dim=256,
        pick_embedding_dim=64,
        max_order_size=20,
        device=None,
    ):
        self.env = env

        self.gamma = float(
            gamma
        )

        self.epsilon = float(
            epsilon
        )

        self.epsilon_min = float(
            epsilon_min
        )

        self.epsilon_decay = float(
            epsilon_decay
        )

        self.batch_size = int(
            batch_size
        )

        self.target_update = int(
            target_update
        )

        self.training_steps = 0

        # Compatibility alias with earlier experiments.
        self.steps_done = 0

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not (
            0.0 <= self.gamma <= 1.0
        ):
            raise ValueError(
                "gamma must lie between 0 and 1."
            )

        if not (
            0.0 <= self.epsilon <= 1.0
        ):
            raise ValueError(
                "epsilon must lie between 0 and 1."
            )

        if not (
            0.0 <= self.epsilon_min <= 1.0
        ):
            raise ValueError(
                "epsilon_min must lie between 0 and 1."
            )

        if not (
            0.0 < self.epsilon_decay <= 1.0
        ):
            raise ValueError(
                "epsilon_decay must lie in (0, 1]."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.target_update <= 0:
            raise ValueError(
                "target_update must be greater than zero."
            )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:

            if (
                hasattr(
                    torch.backends,
                    "mps",
                )
                and torch.backends.mps.is_available()
            ):

                self.device = torch.device(
                    "mps"
                )

            else:

                self.device = torch.device(
                    "cpu"
                )

        else:

            self.device = torch.device(
                device
            )

        # ----------------------------------------------------
        # Networks
        # ----------------------------------------------------

        self.policy_net = SetDQN(
            all_pick_locations=env.all_pick_locations,
            grid_shape=env.grid.shape,
            n_actions=4,
            max_order_size=max_order_size,
            pick_embedding_dim=pick_embedding_dim,
            hidden_dim=hidden_dim,
        ).to(
            self.device
        )

        self.target_net = SetDQN(
            all_pick_locations=env.all_pick_locations,
            grid_shape=env.grid.shape,
            n_actions=4,
            max_order_size=max_order_size,
            pick_embedding_dim=pick_embedding_dim,
            hidden_dim=hidden_dim,
        ).to(
            self.device
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # ----------------------------------------------------
        # Optimiser and replay
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=float(
                lr
            ),
        )

        self.memory = SetReplayBuffer(
            buffer_capacity
        )

        # Optional alias matching terminology used elsewhere.
        self.replay_buffer = (
            self.memory
        )

        self.loss_function = (
            nn.SmoothL1Loss()
        )

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(
        self,
        state,
        eval_mode=False,
    ):
        """
        Select an epsilon-greedy movement action.
        """

        if (
            not eval_mode
            and random.random()
            < self.epsilon
        ):

            return random.randrange(
                4
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            q_values = (
                self.policy_net(
                    state_tensor
                )
            )

        return int(
            torch.argmax(
                q_values
            ).item()
        )

    # ========================================================
    # REPLAY STORAGE
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated=False,
    ):
        """
        Store a transition.

        truncated is intentionally ignored for the Bellman
        terminal mask.

        Rollout code should still stop on:

            terminated or truncated
        """

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            terminated,
        )

    # ========================================================
    # TRAINING STEP
    # ========================================================

    def train_step(
        self,
    ):
        """
        Perform one DQN gradient update.

        Returns
        -------
        float | None
            Loss value, or None if replay memory does not yet
            contain enough samples.
        """

        if len(
            self.memory
        ) < self.batch_size:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminateds,
        ) = self.memory.sample(
            self.batch_size
        )

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        terminateds = torch.as_tensor(
            terminateds,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Current Q-values
        # ----------------------------------------------------

        current_q_values = (
            self.policy_net(
                states
            )
            .gather(
                1,
                actions.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )

        # ----------------------------------------------------
        # Bellman target
        # ----------------------------------------------------

        with torch.no_grad():

            next_q_values = (
                self.target_net(
                    next_states
                )
                .max(
                    dim=1
                )
                .values
            )

            targets = (
                rewards
                + self.gamma
                * next_q_values
                * (
                    1.0
                    - terminateds
                )
            )

        # ----------------------------------------------------
        # Optimisation
        # ----------------------------------------------------

        loss = self.loss_function(
            current_q_values,
            targets,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        self.steps_done = (
            self.training_steps
        )

        # ----------------------------------------------------
        # Target-network update
        # ----------------------------------------------------

        if (
            self.training_steps
            % self.target_update
            == 0
        ):

            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        # ----------------------------------------------------
        # Epsilon decay per gradient update
        # ----------------------------------------------------

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

        return float(
            loss.item()
        )

    # ========================================================
    # Q VALUES
    # ========================================================

    def get_q_values(
        self,
        state,
    ):
        """
        Return greedy Q-values for one raw environment state.
        """

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

        return (
            q_values
            .detach()
            .cpu()
            .numpy()
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path,
    ):
        """
        Save policy-network weights.
        """

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.policy_net.state_dict(),
            path,
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        path,
    ):
        """
        Load policy-network weights and synchronise target net.
        """

        path = Path(
            path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Model checkpoint not found: {path}"
            )

        try:

            state_dict = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )

        except TypeError:

            # Compatibility with older PyTorch versions.
            state_dict = torch.load(
                path,
                map_location=self.device,
            )

        self.policy_net.load_state_dict(
            state_dict
        )

        self.target_net.load_state_dict(
            state_dict
        )

        self.policy_net.train()

        self.target_net.eval()
