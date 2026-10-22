import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          alpha, beta, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // N, offsets % N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(K):
        a = tl.load(A + row * K + k, row < M, other=0)
        b = tl.load(B + k * N + col, row < M, other=0)
        acc += a * b
    old = tl.load(C + offsets, offsets < M * N, other=0)
    tl.store(C + offsets, alpha * acc + beta * old, offsets < M * N)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
          M: int, N: int, K: int, alpha: float, beta: float):
    _gemm[(triton.cdiv(M * N, 128),)](a, b, c, M, N, K, alpha, beta, 128)
