"""Generic CUDA kernel validator for leetgpu.

Parses `extern "C" void solve(...)` from a .cu file, compiles it, runs the
kernel and a user-supplied PyTorch reference side-by-side on identical inputs,
then compares every output tensor for correctness.

Usage:
    python validator.py <solution.cu> --ref=<ref.py> [--DIM=VALUE ...] [options]

Examples:
    # Vector addition
    python validator.py easy/VectorAddition/solution.cu \\
        --ref=refs/vector_add.py --N=1000000

    # Matrix multiplication
    python validator.py "easy/Matrix Multiplication/solution.cu" \\
        --ref=refs/matmul.py --M=1024 --K=1024 --N=1024

ref.py format
-------------
    import torch

    def reference(*, A, B, C, M, K, N, **kwargs):
        \"\"\"
        A, B are input CUDA tensors (1-D flat).
        C is the output CUDA tensor (1-D flat, pre-allocated).
        M, K, N are Python ints.
        \"\"\"
        C[:] = (A.reshape(M, K) @ B.reshape(K, N)).reshape(-1)

    # Optional tolerance overrides
    atol = 1e-4
    rtol = 1e-3
"""

import re
import os
import sys
import subprocess
import ctypes
import argparse
import importlib.util
import torch

# ---------------------------------------------------------------------------
# Type tables (same as benchmark.py)
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {
    "float*":          ("float*",          ctypes.c_void_p),
    "double*":         ("double*",         ctypes.c_void_p),
    "unsigned char*":  ("unsigned char*",  ctypes.c_void_p),
    "unsigned short*": ("unsigned short*", ctypes.c_void_p),
    "unsigned int*":   ("unsigned int*",   ctypes.c_void_p),
    "char*":           ("char*",           ctypes.c_void_p),
    "short*":          ("short*",          ctypes.c_void_p),
    "long*":           ("long*",           ctypes.c_void_p),
    "int*":            ("int*",            ctypes.c_void_p),
    "int":             ("int",             ctypes.c_int),
    "long":            ("long",            ctypes.c_long),
    "size_t":          ("size_t",          ctypes.c_size_t),
    "unsigned int":    ("unsigned int",    ctypes.c_uint),
    "unsigned short":  ("unsigned short",  ctypes.c_ushort),
    "unsigned char":   ("unsigned char",   ctypes.c_ubyte),
    "char":            ("char",            ctypes.c_char),
    "short":           ("short",           ctypes.c_short),
}

DTYPE_MAP = {
    "float*":          torch.float32,
    "double*":         torch.float64,
    "int*":            torch.int32,
    "long*":           torch.int64,
    "short*":          torch.int16,
    "char*":           torch.int8,
    "unsigned char*":  torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*":   getattr(torch, "uint32", torch.int32),
}

INT_TYPES = {"int", "long", "size_t", "unsigned int"}

# ---------------------------------------------------------------------------
# Helpers (shared with benchmark.py)
# ---------------------------------------------------------------------------

def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" void solve(...)` in a .cu file."""
    with open(cu_file, "r") as f:
        content = f.read()

    pattern = r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f'Cannot find \'extern "C" void solve(...)\' in {cu_file}')

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
        token_clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
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
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    return "sm_80"


_STRIP_INCLUDES = re.compile(
    r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE
)


def _preprocess_cu(cu_file: str) -> str:
    with open(cu_file, "r") as f:
        src = f.read()
    cleaned = _STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w") as f:
        f.write(cleaned)
    return tmp


def compile_cu(cu_file: str, output_so: str, arch: str):
    clean_file = _preprocess_cu(cu_file)
    cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC",
        f"-arch={arch}", "-O3", "-o", output_so, clean_file,
    ]
    print(f"[compile] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if clean_file != cu_file and os.path.exists(clean_file):
        os.remove(clean_file)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[compile] -> {output_so}")


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------

def load_reference(ref_file: str):
    """Import a Python file and return its module."""
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference file not found: {ref_file}")
    spec = importlib.util.spec_from_file_location("_leetgpu_ref", ref_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reference"):
        raise AttributeError(
            f"'{ref_file}' must define a `reference(**kwargs)` function."
        )
    return mod


# ---------------------------------------------------------------------------
# Pretty helpers
# ---------------------------------------------------------------------------

def _fmt_row(vals, width=10):
    return "[" + ", ".join(f"{v:>{width}.4f}" for v in vals) + "]"


def _color(text: str, ok: bool) -> str:
    """ANSI color: green for pass, red for fail."""
    code = "\033[92m" if ok else "\033[91m"
    return f"{code}{text}\033[0m"


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def run_validator(
    cu_file: str,
    ref_file: str,
    dim_values: dict,
    ptr_size_override: int,
    arch: str,
    atol: float,
    rtol: float,
    seed: int,
):
    # ---- signature --------------------------------------------------------
    params = parse_solve_signature(cu_file)
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    # ---- compile ----------------------------------------------------------
    so_file = os.path.splitext(cu_file)[0] + ".so"
    compile_cu(cu_file, so_file, arch)
    lib = ctypes.CDLL(so_file)

    # ---- reference --------------------------------------------------------
    ref_mod = load_reference(ref_file)
    ref_fn = ref_mod.reference
    _atol = float(getattr(ref_mod, "atol", atol))
    _rtol = float(getattr(ref_mod, "rtol", rtol))
    print(f"[reference] {ref_file}  (atol={_atol}, rtol={_rtol})\n")

    # ---- determine scalar dim values -------------------------------------
    int_vals: dict[str, int] = {}
    for ptype, pname, _ in params:
        if ptype in INT_TYPES:
            if pname not in dim_values:
                raise ValueError(
                    f"Missing dimension: --{pname}=<value>  (required by kernel signature)"
                )
            int_vals[pname] = dim_values[pname]

    # ---- determine buffer element count ----------------------------------
    vals = list(int_vals.values())
    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif len(vals) == 0:
        ptr_elems = 1024 * 1024
    elif len(vals) == 1:
        ptr_elems = vals[0]
    else:
        sv = sorted(vals, reverse=True)
        ptr_elems = sv[0] * sv[1]
    ptr_elems = min(ptr_elems, 256 * 1024 * 1024)

    # ---- allocate tensors ------------------------------------------------
    # For each pointer param we create two copies (kernel / reference) seeded
    # identically so both start with the same data.
    torch.manual_seed(seed)

    kernel_tensors: dict[str, torch.Tensor] = {}
    ref_tensors:    dict[str, torch.Tensor] = {}
    output_params: list[tuple[str, str]] = []   # (name, ptype) non-const ptrs

    call_args = []
    argtypes   = []

    print("[buffers]")
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if dtype.is_floating_point:
                base = torch.randn(ptr_elems, device="cuda", dtype=dtype)
            else:
                base = torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_()
            kt = base.clone()
            rt = base.clone()
            kernel_tensors[pname] = kt
            ref_tensors[pname]    = rt
            if not is_const:
                output_params.append((pname, ptype))
            call_args.append(ctypes.c_void_p(kt.data_ptr()))
            argtypes.append(ctypes.c_void_p)
            role = "input" if is_const else "output"
            eb   = kt.element_size()
            print(
                f"  {pname:>10s} : {ptype:<16s} [{role:>6s}] "
                f"{ptr_elems} elems  ({ptr_elems * eb / 1024 / 1024:.1f} MB)"
            )
        elif ptype in SUPPORTED_TYPES:
            _, ctype = SUPPORTED_TYPES[ptype]
            val = dim_values[pname]
            call_args.append(ctype(val))
            argtypes.append(ctype)
            print(f"  {pname:>10s} : {ptype:<16s} = {val}")

    lib.solve.restype  = None
    lib.solve.argtypes = argtypes

    if not output_params:
        print("\n[warn] No output tensors detected (all pointer params are const). "
              "Nothing to validate.", file=sys.stderr)
        return True

    # ---- run kernel -------------------------------------------------------
    print("\n[kernel]    running … ", end="", flush=True)
    lib.solve(*call_args)
    torch.cuda.synchronize()
    print("done")

    # ---- run reference ----------------------------------------------------
    ref_kwargs: dict = {}
    for ptype, pname, _ in params:
        if ptype in DTYPE_MAP:
            ref_kwargs[pname] = ref_tensors[pname]
        else:
            ref_kwargs[pname] = dim_values[pname]

    print("[reference] running … ", end="", flush=True)
    ref_fn(**ref_kwargs)
    torch.cuda.synchronize()
    print("done")

    # ---- compare ----------------------------------------------------------
    PREVIEW = 8
    print(f"\n[validate] {len(output_params)} output tensor(s)\n")

    all_pass = True
    for pname, ptype in output_params:
        kt = kernel_tensors[pname].float()
        rt = ref_tensors[pname].float()

        match = torch.allclose(kt, rt, atol=_atol, rtol=_rtol)
        if not match:
            all_pass = False

        max_diff  = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        rel_err   = (
            (kt - rt).abs() / (rt.abs().clamp(min=1e-8))
        ).mean().item()

        status_str = _color("PASS", match) if sys.stdout.isatty() else ("PASS" if match else "FAIL")
        print(f"  [{status_str}]  {pname}  ({ptype})")
        print(f"         max |Δ|   = {max_diff:.6e}")
        print(f"         mean |Δ|  = {mean_diff:.6e}")
        print(f"         mean rel  = {rel_err:.6e}")

        if not match:
            diff_mask = ~torch.isclose(kt, rt, atol=_atol, rtol=_rtol)
            bad_idx   = diff_mask.nonzero(as_tuple=True)[0]
            n_bad     = bad_idx.numel()
            print(f"         mismatches: {n_bad} / {kt.numel()}")
            if n_bad > 0:
                idx = bad_idx[0].item()
                print(f"         first bad   @ idx={idx}:  "
                      f"kernel={kt[idx].item():.6f}  ref={rt[idx].item():.6f}")

        k_vals = kernel_tensors[pname][:PREVIEW].float().cpu().tolist()
        r_vals = ref_tensors[pname][:PREVIEW].float().cpu().tolist()
        print(f"         kernel[:{PREVIEW}] = {_fmt_row(k_vals)}")
        print(f"         ref   [:{PREVIEW}] = {_fmt_row(r_vals)}")
        print()

    # ---- summary ----------------------------------------------------------
    print("=" * 60)
    print(f"  Kernel    : {os.path.basename(cu_file)}")
    print(f"  Reference : {os.path.basename(ref_file)}")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  Arch      : {arch}")
    print(f"  Dims      : {dim_values}")
    print(f"  Buf/ptr   : {ptr_elems} elems")
    print(f"  Tolerance : atol={_atol}  rtol={_rtol}")
    print("-" * 60)
    result_str = "ALL PASS ✓" if all_pass else "FAILED ✗"
    if sys.stdout.isatty():
        result_str = _color(result_str, all_pass)
    print(f"  Result    : {result_str}")
    print("=" * 60)

    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generic CUDA kernel validator for leetgpu",
        epilog=(
            "Dimension args: pass --NAME=VALUE for each int param in solve().\n"
            "ref.py must define `reference(**kwargs)` and may set atol/rtol."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cu_file", help="Path to .cu solution file")
    parser.add_argument("--ref",  required=True, help="Path to reference .py file")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance (default: 1e-4, overridable in ref.py)")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance (default: 1e-3, overridable in ref.py)")
    parser.add_argument("--ptr-size", type=int, default=0,
                        help="Override element count for all pointer buffers")
    parser.add_argument("--arch", type=str, default="",
                        help="GPU arch e.g. sm_90 (auto-detected if omitted)")
    parser.add_argument("--gpu",  type=int, default=0,
                        help="GPU device index (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for input tensors (default: 42)")

    args, unknown = parser.parse_known_args()

    dim_values: dict[str, int] = {}
    for u in unknown:
        if u.startswith("--") and "=" in u:
            key, val = u[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{u}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch()

    ok = run_validator(
        cu_file          = args.cu_file,
        ref_file         = args.ref,
        dim_values       = dim_values,
        ptr_size_override= args.ptr_size,
        arch             = arch,
        atol             = args.atol,
        rtol             = args.rtol,
        seed             = args.seed,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
