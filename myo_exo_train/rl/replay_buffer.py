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
        self.context: torch.Tensor | None = None
        self.next_context: torch.Tensor | None = None
        self.next_external_action: torch.Tensor | None = None

    def enable_assistance(self, context_dim: int, external_action_dim: int) -> None:
        self.context = torch.empty(
            (self.capacity, int(context_dim)), dtype=torch.float32, device=self.device
        )
        self.next_context = torch.empty_like(self.context)
        self.next_external_action = torch.empty(
            (self.capacity, int(external_action_dim)),
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        next_context: torch.Tensor | None = None,
        next_external_action: torch.Tensor | None = None,
    ) -> None:
        assistance_enabled = self.context is not None
        assistance_values = (context, next_context, next_external_action)
        if assistance_enabled != all(value is not None for value in assistance_values):
            raise ValueError("assistance replay fields must be configured and supplied together")
        count = int(obs.shape[0])
        if count >= self.capacity:
            self.obs.copy_(obs[-self.capacity :].detach())
            self.action.copy_(action[-self.capacity :].detach())
            self.reward.copy_(reward[-self.capacity :].detach())
            self.next_obs.copy_(next_obs[-self.capacity :].detach())
            self.done.copy_(done[-self.capacity :].float().detach())
            if assistance_enabled:
                self.context.copy_(context[-self.capacity :].detach())
                self.next_context.copy_(next_context[-self.capacity :].detach())
                self.next_external_action.copy_(
                    next_external_action[-self.capacity :].detach()
                )
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
        if assistance_enabled:
            self.context[destination].copy_(context[:first].detach())
            self.next_context[destination].copy_(next_context[:first].detach())
            self.next_external_action[destination].copy_(
                next_external_action[:first].detach()
            )

        remaining = count - first
        if remaining > 0:
            self.obs[:remaining].copy_(obs[first:].detach())
            self.action[:remaining].copy_(action[first:].detach())
            self.reward[:remaining].copy_(reward[first:].detach())
            self.next_obs[:remaining].copy_(next_obs[first:].detach())
            self.done[:remaining].copy_(done[first:].float().detach())
            if assistance_enabled:
                self.context[:remaining].copy_(context[first:].detach())
                self.next_context[:remaining].copy_(next_context[first:].detach())
                self.next_external_action[:remaining].copy_(
                    next_external_action[first:].detach()
                )

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

    def sample_with_assistance(
        self, batch_size: int
    ) -> tuple[torch.Tensor, ...]:
        if (
            self.context is None
            or self.next_context is None
            or self.next_external_action is None
        ):
            raise RuntimeError("assistance replay storage is not enabled")
        indices = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        return (
            self.obs[indices],
            self.action[indices],
            self.reward[indices],
            self.next_obs[indices],
            self.done[indices],
            self.context[indices],
            self.next_context[indices],
            self.next_external_action[indices],
        )
