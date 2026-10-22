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


def run_case(library, name, host_input, k):
    input_tensor = torch.tensor(host_input, dtype=torch.float32, device="cuda")
    output_tensor = torch.full(
        (k,), float("nan"), dtype=torch.float32, device="cuda"
    )

    library.solve(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output_tensor.data_ptr()),
        ctypes.c_int(input_tensor.numel()),
        ctypes.c_int(k),
    )
    torch.cuda.synchronize()

    expected = torch.topk(input_tensor, k, largest=True, sorted=True)[0]
    got = output_tensor
    passed = torch.equal(got, expected)

    print("\n[case] {}".format(name))
    print("N={}, k={}".format(input_tensor.numel(), k))
    print("expected: {}".format(expected.cpu().tolist()))
    print("got     : {}".format(got.cpu().tolist()))
    print("result  : {}".format("PASS" if passed else "FAIL"))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="solution_sota.cu")
    parser.add_argument("--so", default="solution_sota_test.so")
    parser.add_argument("--arch", default="sm_75")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.so)
    compile_solution(source, output, args.arch)
    solution = load_solution(output)

    # solution_sota.cu launches 256 * 256 threads in its first kernel.
    # Indices separated by 65536 are therefore handled by the same thread.
    # That thread retains only one local_max, so 99.0 is discarded even
    # though it is the true second-largest value.
    total_first_stage_threads = 256 * 256
    adversarial_input = [0.0] * (total_first_stage_threads + 1)
    adversarial_input[0] = 100.0
    adversarial_input[total_first_stage_threads] = 99.0

    passed = run_case(
        solution,
        "two global top values assigned to the same first-stage thread",
        adversarial_input,
        2,
    )

    if passed:
        print("\nUnexpectedly, the adversarial case passed.")
        raise SystemExit(1)

    print("\nHack reproduced: solution_sota.cu does not implement exact Top-K for k > 1.")


if __name__ == "__main__":
    main()
