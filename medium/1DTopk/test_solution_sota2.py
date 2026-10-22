#!/usr/bin/env python3

import argparse
from collections import Counter
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


def validate(library, name, input_tensor, k):
    input_tensor = input_tensor.contiguous().cuda()
    expected = torch.topk(input_tensor, k, largest=True, sorted=True)[0]
    output = torch.full((k,), float("nan"), dtype=torch.float32, device="cuda")

    library.solve(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(input_tensor.numel()),
        ctypes.c_int(k),
    )
    torch.cuda.synchronize()

    mismatch = output != expected
    mismatch_count = int(mismatch.sum().item())
    passed = mismatch_count == 0
    print("\n[case] {}".format(name))
    print("N={}, k={}".format(input_tensor.numel(), k))
    print("result: {}".format("PASS" if passed else "FAIL"))
    print("mismatched output positions: {} / {}".format(mismatch_count, k))

    if not passed:
        expected_values = expected.cpu().tolist()
        output_values = output.cpu().tolist()
        missing = Counter(expected_values) - Counter(output_values)
        extra = Counter(output_values) - Counter(expected_values)
        first = int(torch.nonzero(mismatch)[0].item())
        print("first mismatch at rank {}".format(first))
        print("expected[{}] = {}".format(first, expected[first].item()))
        print("got[{}]      = {}".format(first, output[first].item()))
        print("missing true Top-K values: {}".format(sum(missing.values())))
        print("extra non-Top-K values   : {}".format(sum(extra.values())))
        print("expected[:10] = {}".format(expected_values[:10]))
        print("got[:10]      = {}".format(output_values[:10]))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--arch", default="sm_75")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(here, "solution_sota2.cu")
    shared_library = os.path.join(here, "solution_sota2_test.so")
    compile_solution(source, shared_library, args.arch)
    library = load_solution(shared_library)

    # sota2 launches 128 blocks * 256 threads.  These two values are one
    # grid-stride apart and are therefore both reduced to thread 0's one
    # local_max value.
    stride = 128 * 256
    adversarial = torch.zeros(stride + 1, dtype=torch.float32)
    adversarial[0] = 100.0
    adversarial[stride] = 99.0
    all_passed = validate(library, "same first-stage thread", adversarial, 2)

    if args.large:
        torch.manual_seed(20260904)
        random_input = torch.randn(50000000, dtype=torch.float32, device="cuda")
        all_passed = validate(
            library,
            "leaderboard-size fixed random input",
            random_input,
            100,
        ) and all_passed

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
