#!/usr/bin/env python3
"""Generic CUDA kernel benchmark for leetgpu.

Automatically parses the `extern "C" void solve(...)` signature from a .cu file,
compiles it to a shared library, allocates GPU memory, and benchmarks the kernel.

Usage:
    python benchmark.py <solution.cu> [--DIM=VALUE ...] [options]

Examples:
    # Vector addition: solve(const float* A, const float* B, float* C, int N)
    python benchmark.py easy/VectorAddition/solution.cu --N=1000000

    # Matrix multiplication: solve(const float* A, const float* B, float* C, int M, int N, int K)
    python benchmark.py medium/MatMul/solution.cu --M=1024 --N=1024 --K=1024

    # Reverse array: solve(float* input, int N)
    python benchmark.py easy/ReverseArray/solution.cu --N=1000000

    # Custom buffer size and repeat count
    python benchmark.py solution.cu --N=4096 --warmup=10 --repeat=100

    # Specify pointer sizes explicitly (bytes-per-pointer auto from dims by default)
    python benchmark.py solution.cu --N=1024 --M=1024 --ptr-size=1048576
"""

import re
import os
import sys
import subprocess
import ctypes
import argparse
import torch

SUPPORTED_TYPES = {
    "float*":  ("float*",  ctypes.c_void_p),
    "double*": ("double*", ctypes.c_void_p),
    "int*":    ("int*",    ctypes.c_void_p),
    "int":     ("int",     ctypes.c_int),
    "long":    ("long",    ctypes.c_long),
    "size_t":  ("size_t",  ctypes.c_size_t),
    "unsigned int": ("unsigned int", ctypes.c_uint),
}

DTYPE_MAP = {
    "float*":  torch.float32,
    "double*": torch.float64,
    "int*":    torch.int32,
}


def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" void solve(...)` in a .cu file."""
    with open(cu_file, "r") as f:
        content = f.read()

    pattern = r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(
            f'Cannot find \'extern "C" void solve(...)\' in {cu_file}'
        )

    raw = match.group(1)
    raw = re.sub(r"/\*.*?\*/", "", raw)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue

        is_const = "const" in token
        token_clean = token.replace("const", "").strip()
        token_clean = re.sub(r"\s+", " ", token_clean)

        matched = False
        for key in sorted(SUPPORTED_TYPES.keys(), key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            m = re.match(rf"({base})\s+(\w+)", token_clean)
            if m:
                params.append((key, m.group(2), is_const))
                matched = True
                break

        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")

    return params


def detect_arch() -> str:
    """Auto-detect GPU compute capability and return sm_XX string."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    return "sm_80"


STRIP_INCLUDES = re.compile(
    r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE
)


def preprocess_cu(cu_file: str) -> str:
    """Strip clang-specific includes that break nvcc. Returns path to clean file."""
    with open(cu_file, "r") as f:
        src = f.read()
    cleaned = STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w") as f:
        f.write(cleaned)
    return tmp


def compile_cu(cu_file: str, output_so: str, arch: str):
    """Compile .cu to a shared library."""
    clean_file = preprocess_cu(cu_file)
    cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler", "-fPIC",
        f"-arch={arch}",
        "-O3",
        "-o", output_so,
        clean_file,
    ]
    print(f"[compile] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if clean_file != cu_file and os.path.exists(clean_file):
        os.remove(clean_file)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[compile] -> {output_so}")


def _fmt_vals(vals, width=10):
    """Format a list of numeric values for compact display."""
    return "[" + ", ".join(f"{v:>{width}.4f}" for v in vals) + "]"


def run_benchmark(cu_file, dim_values, warmup, repeat, ptr_size_override, arch):
    params = parse_solve_signature(cu_file)
    sig_str = ", ".join(
        f"{'const ' if c else ''}{t} {n}" for t, n, c in params
    )
    print(f"[signature] solve({sig_str})")

    so_file = os.path.splitext(cu_file)[0] + ".so"
    compile_cu(cu_file, so_file, arch)

    lib = ctypes.CDLL(so_file)

    int_values = []
    for ptype, pname, _ in params:
        if ptype in ("int", "long", "size_t", "unsigned int"):
            if pname not in dim_values:
                raise ValueError(
                    f"Missing dimension: --{pname}=<value>  (required by signature)"
                )
            int_values.append(dim_values[pname])

    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif len(int_values) == 0:
        ptr_elems = 1024 * 1024
    elif len(int_values) == 1:
        ptr_elems = int_values[0]
    else:
        sorted_v = sorted(int_values, reverse=True)
        ptr_elems = sorted_v[0] * sorted_v[1]

    ptr_elems = min(ptr_elems, 256 * 1024 * 1024)

    PREVIEW = 8

    tensor_info = []
    call_args = []
    argtypes = []

    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            t = torch.randn(ptr_elems, device="cuda", dtype=torch.float32).to(dtype)
            role = "input" if is_const else "output"
            tensor_info.append((pname, ptype, role, t))
            call_args.append(ctypes.c_void_p(t.data_ptr()))
            argtypes.append(ctypes.c_void_p)
            elem_bytes = t.element_size()
            print(
                f"  {pname:>10s} : {ptype:<10s} [{role:>6s}] -> {ptr_elems} elems "
                f"({ptr_elems * elem_bytes / 1024 / 1024:.1f} MB)"
            )
        elif ptype in SUPPORTED_TYPES:
            _, ctype = SUPPORTED_TYPES[ptype]
            val = dim_values[pname]
            call_args.append(ctype(val))
            argtypes.append(ctype)
            print(f"  {pname:>10s} : {ptype:<10s} = {val}")

    lib.solve.restype = None
    lib.solve.argtypes = argtypes

    print(f"\n[preview] first {PREVIEW} elements before kernel call:")
    for name, ptype, role, t in tensor_info:
        vals = t[:PREVIEW].cpu().tolist()
        tag = "IN " if role == "input" else "OUT"
        print(f"  {tag} {name:>6s} = {_fmt_vals(vals)}")

    print(f"\n[warmup] {warmup} iterations ...")
    for _ in range(warmup):
        lib.solve(*call_args)
    torch.cuda.synchronize()

    print(f"\n[preview] first {PREVIEW} elements after kernel call:")
    for name, ptype, role, t in tensor_info:
        vals = t[:PREVIEW].cpu().tolist()
        tag = "IN " if role == "input" else "OUT"
        print(f"  {tag} {name:>6s} = {_fmt_vals(vals)}")

    print(f"\n[bench]  {repeat} iterations ...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(repeat):
        start_event.record()
        lib.solve(*call_args)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    avg = sum(times_ms) / len(times_ms)
    med = sorted(times_ms)[len(times_ms) // 2]
    mn = min(times_ms)
    mx = max(times_ms)

    tensors = [t for _, _, _, t in tensor_info]
    total_ptr_bytes = sum(t.nelement() * t.element_size() for t in tensors)

    print()
    print("=" * 55)
    print(f"  Kernel       : {os.path.basename(cu_file)}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Arch         : {arch}")
    print(f"  Dims         : {dim_values}")
    print(f"  Buf/ptr      : {ptr_elems} elems")
    print(f"  Iterations   : {repeat}")
    print("-" * 55)
    print(f"  Average      : {avg:>10.4f} ms")
    print(f"  Median       : {med:>10.4f} ms")
    print(f"  Min          : {mn:>10.4f} ms")
    print(f"  Max          : {mx:>10.4f} ms")
    if avg > 0:
        bw = total_ptr_bytes / (avg / 1000) / 1e9
        print(f"  ~Bandwidth   : {bw:>10.2f} GB/s  (all ptrs, rough)")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Generic CUDA kernel benchmark for leetgpu",
        epilog="Dimension args: pass --NAME=VALUE for each int param in the solve() signature.",
    )
    parser.add_argument("cu_file", help="Path to .cu solution file")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark iterations (default: 100)")
    parser.add_argument("--ptr-size", type=int, default=0,
                        help="Override element count for all pointer buffers")
    parser.add_argument("--arch", type=str, default="",
                        help="GPU arch, e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (default: 0)")

    args, unknown = parser.parse_known_args()

    dim_values = {}
    for u in unknown:
        if u.startswith("--") and "=" in u:
            key, val = u[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{u}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch()

    run_benchmark(args.cu_file, dim_values, args.warmup, args.repeat, args.ptr_size, arch)


if __name__ == "__main__":
    main()

