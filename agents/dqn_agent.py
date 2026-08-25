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

    For the final warehouse:
        input_dim = 98

    Output:
        Q-value for each of the four movement actions:
        0 = up
        1 = down
        2 = left
        3 = right
    """

    def __init__(
        self,
        input_dim,
        n_actions=4,
        hidden_dim=256
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    """
    Stores transitions from previous environment interactions.

    Each transition contains:

        state
        action
        reward
        next_state
        done

    where done should be True when either:
        - the episode terminates successfully, or
        - the episode is truncated.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        # Store copies so later changes to an environment state
        # cannot accidentally change experiences already stored.
        self.buffer.append(
            (
                np.array(state, dtype=np.float32).copy(),
                action,
                reward,
                np.array(next_state, dtype=np.float32).copy(),
                done
            )
        )

    def sample(self, batch_size):
        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.stack(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.stack(next_states),
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        return len(self.buffer)


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
        hidden_dim=256
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

        # State =
        # 2 normalised coordinates
        # + one binary value for every fixed pick location.
        self.input_dim = (
            2 + len(env.all_pick_locations)
        )

        self.n_actions = 4

        # ----------------------------------------------------
        # Policy network
        # ----------------------------------------------------

        self.policy_net = DQN(
            input_dim=self.input_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim
        ).to(self.device)

        # ----------------------------------------------------
        # Target network
        # ----------------------------------------------------

        self.target_net = DQN(
            input_dim=self.input_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim
        ).to(self.device)

        # Start target network with the same weights.
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # ----------------------------------------------------
        # Optimiser
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr
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

        self.steps_done = 0

        # Huber loss is commonly used for DQN because
        # it is less sensitive to very large TD errors.
        self.loss_function = nn.SmoothL1Loss()

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(
        self,
        state,
        eval_mode=False
    ):
        """
        Select an action using epsilon-greedy exploration.

        During training:
            random action with probability epsilon.

        During evaluation:
            always choose the action with the highest Q-value.
        """

        # -----------------------------------------------
        # Exploration
        # -----------------------------------------------

        if (
            not eval_mode
            and random.random() < self.epsilon
        ):
            return random.randrange(
                self.n_actions
            )

        # -----------------------------------------------
        # Exploitation
        # -----------------------------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(
                state_tensor
            )

        action = q_values.argmax(
            dim=1
        ).item()

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
        Add one environment transition to replay memory.
        """

        done = terminated or truncated

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ========================================================
    # TRAINING STEP
    # ========================================================

    def train_step(self):
        """
        Sample a mini-batch from replay memory and perform
        one DQN optimisation step.
        """

        if len(self.memory) < self.batch_size:
            return None

        # -----------------------------------------------
        # Sample replay batch
        # -----------------------------------------------

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # -----------------------------------------------
        # Current Q-values
        # -----------------------------------------------

        all_q_values = self.policy_net(
            states
        )

        q_values = all_q_values.gather(
            1,
            actions
        ).squeeze(1)

        # -----------------------------------------------
        # Target Q-values
        # -----------------------------------------------

        with torch.no_grad():

            next_q_values = (
                self.target_net(next_states)
                .max(dim=1)
                .values
            )

            targets = (
                rewards
                + (1.0 - dones)
                * self.gamma
                * next_q_values
            )

        # -----------------------------------------------
        # Loss
        # -----------------------------------------------

        loss = self.loss_function(
            q_values,
            targets
        )

        # -----------------------------------------------
        # Backpropagation
        # -----------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        # Optional protection against very large gradients.
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        # -----------------------------------------------
        # Update step counter
        # -----------------------------------------------

        self.steps_done += 1

        # -----------------------------------------------
        # Update target network
        # -----------------------------------------------

        if (
            self.steps_done
            % self.target_update
            == 0
        ):
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        # -----------------------------------------------
        # Reduce exploration
        # -----------------------------------------------

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay
        )

        return loss.item()

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save(self, filepath):
        """
        Save the learned policy network.
        """

        torch.save(
            self.policy_net.state_dict(),
            filepath
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(self, filepath):
        """
        Load a previously trained policy.
        """

        self.policy_net.load_state_dict(
            torch.load(
                filepath,
                map_location=self.device
            )
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.policy_net.eval()
        self.target_net.eval()
