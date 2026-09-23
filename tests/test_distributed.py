from copy import deepcopy
import random
import os
from pathlib import Path
import subprocess
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from valen.training.distributed import Distributed, broadcast_trainable, epoch_shard, synchronize_gradients, initialize
from valen.training.checkpoint import capture_rank_state, save_checkpoint, load_checkpoint


def test_epoch_shards_have_no_padding_or_dropped_states():
    for size in (0, 1, 5, 11):
        shards = [epoch_shard(size, 42, 0, rank, 4) for rank in range(4)]
        assert sorted(i for shard in shards for i in shard) == list(range(size))
        assert shards == [epoch_shard(size, 42, 0, rank, 4) for rank in range(4)]


def test_single_gpu_binds_current_device_for_checkpoint_rng(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert initialize("cuda:1").device == "cuda:1"
    assert initialize("cuda").device == "cuda:3"
    assert selected == [1, 3]


def _worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    context = Distributed(rank, 2, "cpu")
    try:
        model = torch.nn.ParameterDict({"text": torch.nn.Parameter(torch.tensor([1. + rank])),
                                       "visual": torch.nn.Parameter(torch.tensor([2. + rank])),
                                       "unused": torch.nn.Parameter(torch.tensor([3. + rank]))})
        broadcast_trainable(model, context)
        reference = deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.1)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.03, weight_decay=.1)
        # Unequal numbers of states/questions. Vision is only used on rank 1.
        batches = [[[[('text', 2.), ('text', -1.)], [('text', 3.)]], [[('visual', -2.)]]],
                   [[[('text', 4.)]], []]]
        def loss(parameters, state):
            return sum((parameters[name] - target).square().sum() for name, target in state) / len(state)
        for batches_by_rank in batches:
            local = batches_by_rank[rank]
            total = int(context.sum([len(local)])[0])
            optimizer.zero_grad(set_to_none=True)
            for state in local:
                (loss(model, state) / total).backward()
            synchronize_gradients(model, context, bucket_bytes=4)
            reference_optimizer.zero_grad(set_to_none=True)
            for states in batches_by_rank:
                for state in states:
                    (loss(reference, state) / total).backward()
            for name in model:
                if reference[name].grad is None:
                    assert model[name].grad is None
                else:
                    torch.testing.assert_close(model[name].grad, reference[name].grad)
            optimizer.step()
            reference_optimizer.step()
            for name in model:
                torch.testing.assert_close(model[name], reference[name])
            assert model['unused'].item() == 3.
        rng = random.Random(100 + rank)
        torch.manual_seed(200 + rank)
        progress = {"step": 2, "cursor": rank + 1, "order": [rank, rank + 2]}
        states = context.gather(capture_rank_state(progress, rng))
        expected_python, expected_torch = rng.random(), torch.rand(3)
        if context.primary:
            save_checkpoint(root, model, optimizer, {"model_path": root}, progress, rng, rank_states=states)
        context.barrier()
        payload = load_checkpoint(root, model, optimizer, rng, rank=rank, world_size=2)
        assert payload['progress'] == progress
        assert rng.random() == expected_python
        torch.testing.assert_close(torch.rand(3), expected_torch)
        with pytest.raises(ValueError, match="world_size"):
            load_checkpoint(root, model, optimizer, rng, rank=rank, world_size=1)
    finally:
        dist.destroy_process_group()


def test_distributed_matches_global_state_average_and_restores_each_rank(tmp_path):
    # A fresh interpreter also avoids inheriting optional media libraries'
    # sys.path changes from earlier processor tests into spawned workers.
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    mp.spawn(_worker, args=(sys.argv[1],), nprocs=2, join=True)
