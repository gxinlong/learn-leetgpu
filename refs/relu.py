"""Reference for: solve(const float* input, float* output, int N)

output[i] = max(0, input[i])  for i in [0, N)
"""
import torch


def reference(*, input, output, N, **kwargs):
    output[:N] = input[:N].clamp(min=0)


atol = 1e-5
rtol = 1e-5
