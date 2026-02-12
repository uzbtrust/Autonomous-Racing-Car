import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any, List

from model import DQN, PrioritizedReplayBuffer


class DQNAgent:

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.995,
        batch_size: int = 128,
        target_update_freq: int = 5,
        buffer_capacity: int = 300_000,
        train_per_step: int = 4,
        model_path: str = "model.pth",
    ) -> None:
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay: float = epsilon_decay
        self.batch_size: int = batch_size
        self.target_update_freq: int = target_update_freq
        self.train_per_step: int = train_per_step
        self.model_path: str = model_path

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[DQN Agent] Using device: {self.device}")

        self.policy_net: DQN = DQN(state_size, action_size).to(self.device)

        self.target_net: DQN = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer: optim.Adam = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.loss_fn: nn.SmoothL1Loss = nn.SmoothL1Loss(reduction='none')

        self.memory: PrioritizedReplayBuffer = PrioritizedReplayBuffer(buffer_capacity)

        self.training_step: int = 0
        self.episode_count: int = 0
        self.high_score: float = float("-inf")

        self.soft_update_tau: float = 0.005

    def select_actions(self, states: List[List[float]], training: bool = True) -> List[int]:
        num = len(states)

        if training and random.random() < self.epsilon:
            return [random.randint(0, self.action_size - 1) for _ in range(num)]

        state_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)
            self.policy_net.train()
        return q_values.argmax(dim=1).tolist()

    def store_transitions(
        self,
        states: List[List[float]],
        actions: List[int],
        rewards: List[float],
        next_states: List[List[float]],
        dones: List[bool],
    ) -> None:
        for i in range(len(states)):
            state_t = torch.tensor(states[i], dtype=torch.float32)
            next_state_t = torch.tensor(next_states[i], dtype=torch.float32)
            self.memory.push(state_t, actions[i], rewards[i], next_state_t, dones[i])

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        total_loss = 0.0
        for _ in range(self.train_per_step):
            sample = self.memory.sample(self.batch_size)
            if sample is None:
                continue

            states, actions, rewards, next_states, dones, indices, weights = sample

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)

            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                target_q = rewards + self.gamma * next_q * (1 - dones)

            td_errors = (current_q - target_q).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

            loss = (self.loss_fn(current_q, target_q) * weights_t).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            self._soft_update_target()

            total_loss += loss.item()
            self.training_step += 1

        return total_loss / self.train_per_step

    def _soft_update_target(self) -> None:
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.soft_update_tau * policy_param.data + (1.0 - self.soft_update_tau) * target_param.data
            )

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("[DQN Agent] Target network hard updated.")

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def step_decay_epsilon(self, num_cars: int) -> None:
        decay = 0.99997 ** num_cars
        self.epsilon = max(self.epsilon_end, self.epsilon * decay)

    def end_episode(self, episode: int, best_score: float) -> None:
        self.episode_count = episode

        if best_score > self.high_score:
            self.high_score = best_score

        if episode % 50 == 0:
            self.save(episode)

    def save(self, episode: Optional[int] = None) -> None:
        checkpoint: Dict[str, Any] = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode": self.episode_count,
            "high_score": self.high_score,
            "training_step": self.training_step,
        }
        torch.save(checkpoint, self.model_path)
        ep_str = f" (Episode {episode})" if episode is not None else ""
        print(f"[DQN Agent] Model saved to {self.model_path}{ep_str}")

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"[DQN Agent] No checkpoint found at {self.model_path}. Starting fresh.")
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.episode_count = checkpoint.get("episode", 0)
            self.high_score = checkpoint.get("high_score", float("-inf"))
            self.training_step = checkpoint.get("training_step", 0)

            print(f"[DQN Agent] Loaded checkpoint from {self.model_path}")
            print(f"  -> Episode: {self.episode_count}, High Score: {self.high_score:.1f}, Epsilon: {self.epsilon:.4f}")
            return True
        except Exception as e:
            print(f"[DQN Agent] Failed to load checkpoint: {e}")
            return False
