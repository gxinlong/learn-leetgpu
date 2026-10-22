#!/usr/bin/env python3
"""Benchmark a Python module exposing solve(tensor: torch.Tensor, size: int, ...).

Examples (run from the leetgpu directory):
    python benchmark_triton.py solution.py --N=1000000
    python benchmark_triton.py gemm.py --M=512 --N=768 --K=256 \
        --alpha=1.0 --beta=0.0 --shape a=MxK --shape b=KxN --shape c=MxN
    python benchmark_triton.py conv.py --input_rows=128 --input_cols=256 \
        --kernel_rows=3 --kernel_cols=3 --shape input=128x256 \
        --shape kernel=3x3 --shape output=126x254

Without --shape, each tensor is a flat buffer sized like benchmark.py: the
product of the two largest positive int arguments, one int, or 1048576 elements.
This is a capacity heuristic, NOT inference of an operation's shape/semantics.
Use --shape NAME=DIMxDIM (or --ptr-size ELEMENTS for all flat buffers) whenever
the kernel needs a particular shape or capacity. Shapes may use int arg names.

Timing uses CUDA events on the current PyTorch stream around solve(). JIT and
warmup are excluded; host launch gaps can be included. solve() must use that
stream or join any additional streams before returning. Repeated calls reuse
the same buffers, including in-place updates. Previews are not correctness tests.
Compatible with Python 3.6 / PyTorch 1.6; also usable with newer environments.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import statistics
import sys

import torch


DTYPES = {
    "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    "int8": torch.int8, "uint8": torch.uint8, "int16": torch.int16,
    "int32": torch.int32, "int64": torch.int64, "bool": torch.bool,
}
if hasattr(torch, "bfloat16"):
    DTYPES["bfloat16"] = torch.bfloat16


def load_solution(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path) or not path.endswith(".py"):
        raise ValueError("Expected an existing .py solution file: {}".format(path))
    name = "_leetgpu_solution_" + hashlib.sha256(path.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Triton JIT and sibling imports need a real module and source filename.
    sys.modules[name] = module
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise ValueError("{} must define a callable solve()".format(path))
    return solve


def parameter_kind(param):
    annotation = param.annotation
    if annotation is torch.Tensor:
        return "tensor"
    if annotation in (int, float, bool):
        return annotation.__name__
    if isinstance(annotation, str):
        name = annotation.strip("'\"").split(".")[-1]
        if name == "Tensor":
            return "tensor"
        if name in ("int", "float", "bool"):
            return name
    if annotation is inspect.Parameter.empty and type(param.default) in (int, float, bool):
        return type(param.default).__name__
    raise ValueError("Parameter {!r} needs a torch.Tensor, int, float or bool annotation".format(param.name))


def assignments(items, label):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("{} expects NAME=VALUE, got {!r}".format(label, item))
        name, value = item.split("=", 1)
        if not name or not value or name in result:
            raise ValueError("Empty or duplicate {} assignment: {!r}".format(label, item))
        result[name] = value
    return result


def parse_scalar_options(unknown, explicit):
    items = list(explicit)
    index = 0
    while index < len(unknown):
        token = unknown[index]
        if not token.startswith("--") or token == "--":
            raise ValueError("Unexpected argument: {!r}; use --NAME=VALUE".format(token))
        token = token[2:]
        if "=" not in token:
            index += 1
            if index >= len(unknown) or unknown[index].startswith("--"):
                raise ValueError("Missing value for --{}".format(token))
            token += "=" + unknown[index]
        items.append(token)
        index += 1
    return assignments(items, "scalar")


def scalar_value(raw, kind, name):
    try:
        if kind == "bool":
            value = str(raw).lower()
            if value not in ("true", "false", "1", "0"):
                raise ValueError("expected true/false/1/0")
            return value in ("true", "1")
        if kind == "int":
            return int(raw)
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("expected a finite float")
        return value
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid {} value for --{}: {!r} ({})".format(kind, name, raw, exc))


def parse_shape(text, scalars):
    # Commas also work for parameter names containing the letter x.
    tokens = text.split(",") if "," in text else text.split("x")
    shape = []
    for token in tokens:
        token = token.strip()
        value = scalars.get(token)
        if value is None and re.fullmatch(r"[0-9]+", token):
            value = int(token)
        if type(value) is not int or value <= 0:
            raise ValueError("Invalid shape dimension {!r}; use positive ints or int argument names".format(token))
        shape.append(value)
    return tuple(shape)


def prepare_specs(solve, raw_scalars, raw_shapes, raw_dtypes, dtype, ptr_size):
    signature = inspect.signature(solve)
    params = list(signature.parameters.values())
    kinds = {}
    for param in params:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError("Variadic solve(*args, **kwargs) is not supported; name and annotate each parameter")
        kinds[param.name] = parameter_kind(param)
    tensor_names = {name for name, kind in kinds.items() if kind == "tensor"}
    for name in raw_scalars:
        if name not in kinds or name in tensor_names:
            raise ValueError("Unknown scalar parameter: --{}".format(name))
    for label, mapping in (("--shape", raw_shapes), ("--tensor-dtype", raw_dtypes)):
        for name in mapping:
            if name not in tensor_names:
                raise ValueError("{} names an unknown tensor: {}".format(label, name))
    scalars = {}
    for param in params:
        kind = kinds[param.name]
        if kind == "tensor":
            continue
        raw = raw_scalars.get(param.name, param.default)
        if raw is inspect.Parameter.empty:
            raise ValueError("Missing scalar: --{}=<{}>".format(param.name, kind))
        scalars[param.name] = scalar_value(raw, kind, param.name)
    dims = sorted((v for v in scalars.values() if type(v) is int and v > 0), reverse=True)
    capacity = ptr_size or (dims[0] * dims[1] if len(dims) > 1 else dims[0] if dims else 1048576)
    specs = {}
    for param in params:
        if param.name not in tensor_names:
            continue
        name = param.name
        shape = parse_shape(raw_shapes[name], scalars) if name in raw_shapes else (capacity,)
        dtype_name = raw_dtypes.get(name, dtype)
        if dtype_name not in DTYPES:
            raise ValueError("Unknown dtype for {}: {} (choose {})".format(name, dtype_name, ", ".join(sorted(DTYPES))))
        specs[name] = (shape, dtype_name)
    return signature, scalars, specs


def allocate_tensors(specs, gpu):
    tensors = {}
    for name, (shape, dtype_name) in specs.items():
        dtype = DTYPES[dtype_name]
        tensor = torch.empty(shape, device="cuda:{}".format(gpu), dtype=dtype)
        if dtype.is_floating_point:
            tensor.normal_()
        elif dtype == torch.bool:
            tensor.fill_(False)
        else:
            # Small nonnegative integers; callers needing meaningful indices
            # or structured input data must adapt allocation for their task.
            tensor.random_(0, 8)
        tensors[name] = tensor
        print("  {:>14s}: {} {} ({} elements, {:.2f} MiB)".format(
            name, dtype_name, tuple(shape), tensor.numel(), tensor.numel() * tensor.element_size() / 1048576))
    return tensors


def make_call(solve, signature, values):
    positional, keyword = [], {}
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword[param.name] = values[param.name]
        else:
            positional.append(values[param.name])
    return lambda: solve(*positional, **keyword)


def preview(tensors, count, label):
    if count:
        print("[preview] {} (not a correctness check)".format(label))
        for name, tensor in tensors.items():
            print("  {} = {}".format(name, tensor.reshape(-1)[:count].cpu().tolist()))


def run_benchmark(args, raw_scalars):
    import triton
    solve = load_solution(args.py_file)
    signature, scalars, specs = prepare_specs(
        solve, raw_scalars, assignments(args.shape, "--shape"),
        assignments(args.tensor_dtype, "--tensor-dtype"), args.dtype, args.ptr_size)
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in {}".format(sys.executable))
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError("Invalid --gpu {}; {} visible devices".format(args.gpu, torch.cuda.device_count()))
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    print("[environment] Python {} | Torch {} | Triton {} | CUDA {}".format(
        sys.version.split()[0], torch.__version__, triton.__version__, torch.version.cuda))
    print("[signature] solve{}".format(signature))
    print("[scalars] {}".format(scalars))
    if any(name not in assignments(args.shape, "--shape") for name in specs):
        print("[allocation] Unspecified shapes use flat buffers (capacity heuristic); override with --shape NAME=...")
    tensors = allocate_tensors(specs, args.gpu)
    values = dict(scalars)
    values.update(tensors)
    call = make_call(solve, signature, values)
    # Compilation, autotuning, previews and tensor creation are all untimed.
    with torch.no_grad():
        preview(tensors, args.preview, "before first call")
        print("[compile] First solve() call (JIT/autotune, excluded from timing) ...", flush=True)
        result = call()
        torch.cuda.synchronize()
        preview(tensors, args.preview, "after first call")
        if result is not None:
            print("[return] solve() returned {}; timing includes the entire solve() call".format(type(result).__name__))
        del result
        print("[warmup] {} iterations (buffers reused, including in-place updates)".format(args.warmup), flush=True)
        for _ in range(args.warmup):
            result = call()
        torch.cuda.synchronize()
        result = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Materialize both CUDA events before collecting samples.
        start.record()
        end.record()
        end.synchronize()
        print("[bench] {} iterations, CUDA events around solve()".format(args.repeat), flush=True)
        times_ms = []
        for _ in range(args.repeat):
            start.record()
            result = call()
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
            result = None
    report = {
        "file": os.path.abspath(args.py_file), "gpu_index": args.gpu,
        "gpu": torch.cuda.get_device_name(args.gpu),
        "compute_capability": list(torch.cuda.get_device_capability(args.gpu)),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "scalars": scalars,
        "tensors": {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in specs.items()},
        "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
        "timing": "current-stream CUDA events around solve; buffers reused",
        "correctness_checked": False, "times_ms": times_ms,
        "average_ms": statistics.mean(times_ms), "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms), "max_ms": max(times_ms),
    }
    print("\n" + "=" * 60)
    print("  File         : {}".format(report["file"]))
    print("  GPU          : {} (index {}, sm_{}{})".format(report["gpu"], args.gpu, *report["compute_capability"]))
    print("  Iterations   : {}".format(args.repeat))
    for label, key in (("Average", "average_ms"), ("Median", "median_ms"), ("Min", "min_ms"), ("Max", "max_ms")):
        print("  {:<13s}: {:10.6f} ms ({:.3f} us)".format(label, report[key], report[key] * 1000))
    print("  Correctness  : not checked; previews only")
    print("  Timing       : current-stream CUDA events; may include host launch gaps")
    print("=" * 60)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print("[json] {}".format(os.path.abspath(args.json)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("py_file", help="Path to a Python file defining solve()")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0, help="Visible GPU index (default: 0)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--ptr-size", type=int, default=0, help="Flat element count for unspecified shapes (not bytes)")
    parser.add_argument("--shape", action="append", default=[], metavar="NAME=DIMxDIM", help="Per-tensor shape; repeat for multiple tensors; int arg names allowed")
    parser.add_argument("--tensor-dtype", action="append", default=[], metavar="NAME=DTYPE")
    parser.add_argument("--arg", action="append", default=[], metavar="NAME=VALUE", help="Scalar argument, also usable when its name conflicts with a benchmark option")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=8, help="Preview first N tensor elements; 0 disables")
    parser.add_argument("--json", metavar="PATH", help="Save timings and configuration as JSON")
    args, unknown = parser.parse_known_args(argv)
    if args.warmup < 0 or args.repeat <= 0 or args.ptr_size < 0 or args.preview < 0:
        parser.error("warmup, ptr-size and preview must be >= 0; repeat must be > 0")
    try:
        return run_benchmark(args, parse_scalar_options(unknown, args.arg))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
