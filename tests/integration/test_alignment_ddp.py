"""The differentiable feature gather must match a real global-batch gradient."""
import os
from pathlib import Path
import subprocess
import sys


def test_distributed_itc_gradient_matches_global_batch(tmp_path):
    script = tmp_path / 'check.py'
    script.write_text('''
import torch
import torch.distributed as dist
from torch.nn import functional as F
from valen.pretraining.model import contrastive_loss
dist.init_process_group('gloo')
torch.set_num_threads(1)
torch.manual_seed(3)
x, y = torch.randn(8, 4), torch.randn(8, 4)
w = torch.randn(4, 3, requires_grad=True)
reference = w.detach().clone().requires_grad_(True)
rank = dist.get_rank()
sl = slice(rank * 4, (rank + 1) * 4)
groups = torch.arange(8)
loss, _ = contrastive_loss(F.normalize(x[sl] @ w, dim=-1), F.normalize(y[sl] @ w, dim=-1), groups[sl], .1)
loss.backward()
dist.all_reduce(w.grad)
w.grad /= 2
expected, _ = contrastive_loss(F.normalize(x @ reference, dim=-1), F.normalize(y @ reference, dim=-1), groups, .1, distributed=False)
expected.backward()
torch.testing.assert_close(w.grad, reference.grad, rtol=2e-5, atol=2e-6)
dist.destroy_process_group()
''')
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1')
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                    str(script)], env=environment, check=True, capture_output=True, text=True, timeout=240)
