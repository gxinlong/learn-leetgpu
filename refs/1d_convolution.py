"""Reference for: solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size)

output = conv1d(input, kernel, stride=1)  →  output_size = input_size - kernel_size + 1
"""
import torch


def reference(*, input, kernel, output, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    inp = input[:input_size].reshape(1, 1, input_size)
    ker = kernel[:kernel_size].reshape(1, 1, kernel_size)
    result = torch.nn.functional.conv1d(inp, ker, stride=1)
    output[:output_size] = result.reshape(-1)
