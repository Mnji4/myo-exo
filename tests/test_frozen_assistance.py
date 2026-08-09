import torch

from myo_exo_train.rl.frozen_assistance import (
    ExoConditionedHuman,
    RecurrentExoPolicy,
)
from myo_exo_train.rl.replay_buffer import ReplayBuffer


def test_recurrent_exo_enforces_command_slew() -> None:
    policy = RecurrentExoPolicy(6, 8, 1, 0.12, "delta")
    sensor = torch.randn(4, 6)
    previous = torch.zeros(4, 2)
    action, hidden = policy.step(sensor, previous, None)
    assert action.shape == (4, 2)
    assert hidden.shape == (1, 4, 8)
    assert torch.all(torch.abs(action - previous) <= 0.120001)


def test_conditioned_human_keeps_gradient_to_base_action() -> None:
    model = ExoConditionedHuman(12, 4, 8, 6, True, False)
    obs = torch.randn(3, 12)
    base = torch.randn(3, 4, requires_grad=True)
    context = torch.randn(3, 6)
    model(obs, base, context).sum().backward()
    assert base.grad is not None
    assert torch.isfinite(base.grad).all()


def test_replay_assistance_fields_share_sample_indices() -> None:
    replay = ReplayBuffer(16, 3, 4, torch.device("cpu"))
    replay.enable_assistance(5, 2)
    marker = torch.arange(8, dtype=torch.float32)
    replay.add(
        marker[:, None].repeat(1, 3),
        marker[:, None].repeat(1, 4),
        marker,
        marker[:, None].repeat(1, 3),
        torch.zeros(8),
        context=marker[:, None].repeat(1, 5),
        next_context=marker[:, None].repeat(1, 5),
        next_external_action=marker[:, None].repeat(1, 2),
    )
    batch = replay.sample_with_assistance(32)
    for value in batch[1:4] + batch[5:]:
        assert torch.equal(batch[0][:, 0], value[:, 0] if value.ndim == 2 else value)
