#!/usr/bin/env python3

import argparse
import ctypes
import os
import subprocess

import torch


def compile_solution(source, output, arch):
    command = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "-arch={}".format(arch),
        "-o",
        output,
        source,
    ]
    print("[compile] {}".format(" ".join(command)))
    subprocess.check_call(command)


def load_solution(path):
    library = ctypes.CDLL(path)
    library.solve.restype = None
    library.solve.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return library


def check_case(library, name, input_tensor, k):
    input_tensor = input_tensor.contiguous().cuda()
    output = torch.full((k,), float("nan"), device="cuda", dtype=torch.float32)
    expected = torch.topk(input_tensor, k, largest=True, sorted=True)[0]

    library.solve(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(input_tensor.numel()),
        ctypes.c_int(k),
    )
    torch.cuda.synchronize()

    passed = torch.equal(output, expected)
    max_diff = 0.0 if passed else (output - expected).abs().max().item()
    print(
        "{:<28s} N={:<7d} k={:<4d} {} max_diff={}".format(
            name,
            input_tensor.numel(),
            k,
            "PASS" if passed else "FAIL",
            max_diff,
        )
    )
    if not passed:
        print("expected[:16] = {}".format(expected[:16].cpu().tolist()))
        print("got[:16]      = {}".format(output[:16].cpu().tolist()))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--large",
        action="store_true",
        help="also validate the N=50,000,000, k=99 fallback path",
    )
    parser.add_argument("--arch", default="sm_75")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(here, "solution.cu")
    shared_library = os.path.join(here, "solution_fallback_test.so")
    compile_solution(source, shared_library, args.arch)
    library = load_solution(shared_library)
    torch.manual_seed(20260904)

    cases = [
        (
            "statement example 1",
            torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0]),
            3,
        ),
        (
            "statement example 2",
            torch.tensor([7.2, -1.0, 3.3, 8.8, 2.2]),
            2,
        ),
        (
            "duplicates and infinities",
            torch.tensor(
                [
                    float("-inf"),
                    -8.0,
                    3.0,
                    3.0,
                    -0.0,
                    0.0,
                    19.0,
                    float("inf"),
                    19.0,
                ]
            ),
            7,
        ),
        ("k equals N", torch.randn(257), 257),
        ("single chunk k=99", torch.randn(4095), 99),
        ("chunk boundary k=101", torch.randn(4097), 101),
        ("multi chunk k=17", torch.randn(65537), 17),
        ("optimized k=100", torch.randn(65537), 100),
    ]
    if args.large:
        cases.append(("large fallback k=99", torch.randn(50000000), 99))

    passed = True
    for name, values, k in cases:
        passed = check_case(library, name, values, k) and passed

    if not passed:
        raise SystemExit(1)
    print("All fallback and optimized-path checks passed.")


if __name__ == "__main__":
    main()
