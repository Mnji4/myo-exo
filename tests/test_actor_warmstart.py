import torch
import torch.nn as nn

from myo_exo_train.checkpoint import load_actor_warmstart_state_dict


class Actor(nn.Module):
    def __init__(self, actions: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.mean = nn.Linear(4, actions)
        self.logstd = nn.Linear(4, actions)


def test_actor_warmstart_crops_action_heads() -> None:
    source = Actor(4)
    target = Actor(2)
    with torch.no_grad():
        source.encoder.weight.fill_(1.0)
        source.mean.weight.copy_(torch.arange(16).reshape(4, 4))
        source.logstd.bias.copy_(torch.arange(4))

    metadata = load_actor_warmstart_state_dict(target, source.state_dict(), action_dims=2)

    assert torch.equal(target.encoder.weight, source.encoder.weight)
    assert torch.equal(target.mean.weight, source.mean.weight[:2])
    assert torch.equal(target.logstd.bias, source.logstd.bias[:2])
    assert metadata["cropped_key_count"] == 4
