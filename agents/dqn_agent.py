import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# DQN NETWORK
# ============================================================

class DQN(nn.Module):
    """
    Fully connected Deep Q-Network.

    Input:
        [normalised_row,
         normalised_column,
         remaining_pick_1,
         ...
         remaining_pick_96]

    For the fixed warehouse:
        input_dim = 98

    Output:
        Q-value for each movement action:

        0 = up
        1 = down
        2 = left
        3 = right
    """

    def __init__(
        self,
        input_dim,
        n_actions=4,
        hidden_dim=256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                n_actions,
            ),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    """
    Store transitions from previous environment interactions.

    Each transition contains:

        state
        action
        reward
        next_state
        terminated

    Important
    ---------
    Only genuine environment termination is stored in the
    Bellman terminal mask.

    In this project:

        terminated = True
            All required picks have been collected and the
            picker has returned to the depot.

        truncated = True
            The artificial max_steps limit was reached.

    A truncated transition is NOT treated as terminal for
    the Bellman target. The agent therefore continues to
    bootstrap from next_state for truncated transitions.
    """

    def __init__(
        self,
        capacity=10000,
    ):
        self.buffer = deque(
            maxlen=capacity
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

        Copies are used so that later modifications to an
        environment state cannot alter replay history.
        """

        self.buffer.append(
            (
                np.array(
                    state,
                    dtype=np.float32,
                ).copy(),

                int(action),

                float(reward),

                np.array(
                    next_state,
                    dtype=np.float32,
                ).copy(),

                bool(terminated),
            )
        )

    def sample(
        self,
        batch_size,
    ):
        """
        Sample one random mini-batch.
        """

        batch = random.sample(
            self.buffer,
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

        states = torch.tensor(
            np.stack(
                states
            ),
            dtype=torch.float32,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
        ).unsqueeze(
            1
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
        )

        next_states = torch.tensor(
            np.stack(
                next_states
            ),
            dtype=torch.float32,
        )

        terminateds = torch.tensor(
            terminateds,
            dtype=torch.float32,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            terminateds,
        )

    def __len__(self):
        return len(
            self.buffer
        )


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(
        self,
        env,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        batch_size=64,
        buffer_capacity=10000,
        target_update=1000,
        hidden_dim=256,
    ):

        self.env = env

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # State and action dimensions
        # ----------------------------------------------------

        self.input_dim = (
            2
            + len(
                env.all_pick_locations
            )
        )

        self.n_actions = 4

        # ----------------------------------------------------
        # Policy network
        # ----------------------------------------------------

        self.policy_net = DQN(
            input_dim=self.input_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim,
        ).to(
            self.device
        )

        # ----------------------------------------------------
        # Target network
        # ----------------------------------------------------

        self.target_net = DQN(
            input_dim=self.input_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim,
        ).to(
            self.device
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # ----------------------------------------------------
        # Optimiser
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
        )

        # ----------------------------------------------------
        # Replay memory
        # ----------------------------------------------------

        self.memory = ReplayBuffer(
            capacity=buffer_capacity
        )

        # ----------------------------------------------------
        # Hyperparameters
        # ----------------------------------------------------

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.target_update = target_update

        # Number of gradient updates performed.
        self.steps_done = 0

        # Huber loss is commonly used for DQN because it is
        # less sensitive to unusually large TD errors.
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
        Select an action using epsilon-greedy exploration.

        Training
        --------
        Random action with probability epsilon.

        Evaluation
        ----------
        If eval_mode=True, always choose the action with
        the largest policy-network Q-value.
        """

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if (
            not eval_mode
            and random.random()
            < self.epsilon
        ):

            return random.randrange(
                self.n_actions
            )

        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(
            0
        )

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

        action = int(
            q_values.argmax(
                dim=1
            ).item()
        )

        return action

    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated
    ):
        """
        Store one environment transition.

        terminated=True:
            genuine task completion, so Bellman
            bootstrapping must stop.

        truncated=True:
            artificial max_steps cutoff, so the
            rollout stops but Bellman bootstrapping
            should continue.
        """

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            terminated
        )

    # ========================================================
    # TRAINING STEP
    # ========================================================

    def train_step(self):
        """
        Sample a replay mini-batch and perform one DQN
        optimisation step.

        Bellman target:

            y =
                r
                + gamma
                * (1 - terminated)
                * max_a' Q_target(s', a')

        Truncated transitions therefore continue to bootstrap.
        """

        if (
            len(
                self.memory
            )
            < self.batch_size
        ):
            return None

        # ----------------------------------------------------
        # Sample replay batch
        # ----------------------------------------------------

        (
            states,
            actions,
            rewards,
            next_states,
            terminateds,
        ) = self.memory.sample(
            self.batch_size
        )

        states = states.to(
            self.device
        )

        actions = actions.to(
            self.device
        )

        rewards = rewards.to(
            self.device
        )

        next_states = next_states.to(
            self.device
        )

        terminateds = terminateds.to(
            self.device
        )

        # ----------------------------------------------------
        # Current Q-values
        # ----------------------------------------------------

        self.policy_net.train()

        all_q_values = self.policy_net(
            states
        )

        q_values = all_q_values.gather(
            1,
            actions,
        ).squeeze(
            1
        )

        # ----------------------------------------------------
        # Target Q-values
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
                + (
                    1.0
                    - terminateds
                )
                * self.gamma
                * next_q_values
            )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = self.loss_function(
            q_values,
            targets,
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        # ----------------------------------------------------
        # Training-update counter
        # ----------------------------------------------------

        self.steps_done += 1

        # ----------------------------------------------------
        # Target-network update
        # ----------------------------------------------------

        if (
            self.steps_done
            % self.target_update
            == 0
        ):

            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

            self.target_net.eval()

        # ----------------------------------------------------
        # Exploration decay
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
    # SAVE MODEL
    # ========================================================

    def save(
        self,
        filepath,
    ):
        """
        Save the policy-network parameters.
        """

        torch.save(
            self.policy_net.state_dict(),
            filepath,
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(
        self,
        filepath,
    ):
        """
        Load previously saved policy-network parameters and
        synchronise the target network.
        """

        state_dict = torch.load(
            filepath,
            map_location=self.device,
        )

        self.policy_net.load_state_dict(
            state_dict
        )

        self.target_net.load_state_dict(
            state_dict
        )

        self.policy_net.eval()

        self.target_net.eval()
