"""Small real Triton kernel for exercising benchmark_triton.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _add(A, B, C, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offsets, offsets < N, other=0)
    b = tl.load(B + offsets, offsets < N, other=0)
    tl.store(C + offsets, a + b, offsets < N)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    _add[(triton.cdiv(N, 256),)](a, b, c, N, 256)
