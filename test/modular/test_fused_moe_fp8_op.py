"""fused_experts_fp8 is one custom op to dynamo, with shapes known without a GPU."""
import sys
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mstar.utils.fused_moe  # noqa: E402, F401 -- registers mstar::fused_experts_fp8


def _fake_args(tokens: int):
    x = torch.empty(tokens, 256, dtype=torch.bfloat16)
    w1 = torch.empty(4, 256, 256, dtype=torch.uint8)
    w2 = torch.empty(4, 256, 128, dtype=torch.uint8)
    s1, s2 = torch.empty(4, 2, 2), torch.empty(4, 2, 1)
    return x, w1, w2, s1, s2, torch.empty(tokens, 2), torch.empty(tokens, 2, dtype=torch.int64)


@pytest.mark.parametrize("reduce_results", [True, False])
def test_fake_impl_shapes(reduce_results):
    # the op itself: eager, fused_experts_fp8 calls the implementation
    with FakeTensorMode():
        out = torch.ops.mstar.fused_experts_fp8(*_fake_args(37), [128, 128], "silu", reduce_results, None)
    assert out.shape == ((37, 256) if reduce_results else (37, 2, 256))
    assert out.dtype == torch.bfloat16
