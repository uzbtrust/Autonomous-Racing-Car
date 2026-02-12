import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DQN(nn.Module):

    def __init__(self, state_size: int, action_size: int) -> None:
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.LayerNorm(128)

        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class SumTree:

    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.tree: np.ndarray = np.zeros(2 * capacity - 1)
        self.data: list = [None] * capacity
        self.write: int = 0
        self.n_entries: int = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: tuple) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, tuple]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:

    def __init__(self, capacity: int = 200_000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 0.01) -> None:
        self.tree: SumTree = SumTree(capacity)
        self.capacity: int = capacity
        self.alpha: float = alpha
        self.beta: float = beta
        self.beta_increment: float = beta_increment
        self.epsilon: float = epsilon
        self.max_priority: float = 1.0

    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool) -> None:
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor, list, np.ndarray]:
        indices: list = []
        priorities: list = []
        batch: list = []

        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None:
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
                if data is None:
                    continue
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        if len(batch) == 0:
            return None

        priorities_arr = np.array(priorities, dtype=np.float32)
        probs = priorities_arr / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices: list, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
