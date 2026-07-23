"""GPU replay storage used by SAC."""

from __future__ import annotations

import torch

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.action = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.reward = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.done = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.next_idx = 0
        self.size = 0

    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        count = int(obs.shape[0])
        if count >= self.capacity:
            self.obs.copy_(obs[-self.capacity :].detach())
            self.action.copy_(action[-self.capacity :].detach())
            self.reward.copy_(reward[-self.capacity :].detach())
            self.next_obs.copy_(next_obs[-self.capacity :].detach())
            self.done.copy_(done[-self.capacity :].float().detach())
            self.next_idx = 0
            self.size = self.capacity
            return

        first = min(count, self.capacity - self.next_idx)
        destination = slice(self.next_idx, self.next_idx + first)
        self.obs[destination].copy_(obs[:first].detach())
        self.action[destination].copy_(action[:first].detach())
        self.reward[destination].copy_(reward[:first].detach())
        self.next_obs[destination].copy_(next_obs[:first].detach())
        self.done[destination].copy_(done[:first].float().detach())

        remaining = count - first
        if remaining > 0:
            self.obs[:remaining].copy_(obs[first:].detach())
            self.action[:remaining].copy_(action[first:].detach())
            self.reward[:remaining].copy_(reward[first:].detach())
            self.next_obs[:remaining].copy_(next_obs[first:].detach())
            self.done[:remaining].copy_(done[first:].float().detach())

        self.next_idx = (self.next_idx + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        return (
            self.obs[indices],
            self.action[indices],
            self.reward[indices],
            self.next_obs[indices],
            self.done[indices],
        )
