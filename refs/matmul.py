"""Reference for: solve(const float* A, const float* B, float* C, int M, int K, int N)

C = A (M×K) @ B (K×N)  →  C is stored row-major, flat.
"""
import torch


def reference(*, A, B, C, M, K, N, **kwargs):
    C[:M * N] = (A[:M * K].reshape(M, K) @ B[:K * N].reshape(K, N)).reshape(-1)


atol = 1e-3
rtol = 1e-3
