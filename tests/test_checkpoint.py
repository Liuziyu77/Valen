import random
import torch
import pytest
from visionjev.training.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_restores_optimizer_rng_and_frozen_parameters_stay_untouched(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    optimizer = torch.optim.AdamW(model[1].parameters(), lr=.01)
    x = torch.randn(2, 3)
    model(x).square().mean().backward()
    optimizer.step()
    rng = random.Random(123)
    config = {"model_path": str(tmp_path / "base")}
    save_checkpoint(tmp_path, model, optimizer, config, {"step": 1}, rng)
    expected_rng = rng.random()
    expected_torch = torch.rand(2)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model[1].weight.zero_()
    loaded = load_checkpoint(tmp_path, model, optimizer, rng)
    assert set(loaded["weights"]) == {"1.weight", "1.bias"}
    assert loaded["progress"]["step"] == 1
    assert rng.random() == expected_rng
    torch.testing.assert_close(torch.rand(2), expected_torch)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])
    assert optimizer.state[model[1].weight]["step"] == 1
    model[0].requires_grad_(True)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(tmp_path, model)
