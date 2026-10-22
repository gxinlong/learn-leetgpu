import torch
import triton
import triton.language as tl


@triton.jit
def _conv(Input, Kernel, Output, IR: tl.constexpr, IC: tl.constexpr,
          KR: tl.constexpr, KC: tl.constexpr, BLOCK: tl.constexpr):
    OR = IR - KR + 1
    OC = IC - KC + 1
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // OC, offsets % OC
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kr in range(KR):
        for kc in range(KC):
            value = tl.load(Input + (row + kr) * IC + col + kc, row < OR, other=0)
            weight = tl.load(Kernel + kr * KC + kc)
            acc += value * weight
    tl.store(Output + offsets, acc, offsets < OR * OC)


def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    count = (input_rows - kernel_rows + 1) * (input_cols - kernel_cols + 1)
    _conv[(triton.cdiv(count, 128),)](
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, 128)
