# 命令

benckmark 命令：

```
benchmark.py easy/Matrix\ Multiplication/solution.cu --M=8192 --K=6144 --N=4096 --repeat=20
```

- 自动编译 cu 文件
- 自动捕获参数名后创建对应大小的 Tensor, M、K、N 等随 kernel extern 包装函数变化, 多个参数时会尽量构造最大的二维矩阵
- 不测试正确性, 只测试调用性能

比如上面的命令对应下面的签名：

```
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
```


ncu 命令：

```
ncu --target-processes all \
        --profile-from-start on \
        --launch-skip 20 \
        --launch-count 2 \
        --set full \
        -o profile_log -f \
        $(which python) benchmark.py easy/Matrix\ Multiplication/solution_register_prefetch.cu --M=8192 --K=6144 --N=4096 --repeat=20
```

## Triton benchmark

`benchmark_triton.py` 加载 Python 文件中的 `solve(...)`，依据参数注解创建 GPU
Tensor，并用 CUDA Event 测量整个 `solve` 调用。支持 `torch.Tensor`、`int`、
`float`、`bool` 参数；标量通过 `--参数名=值` 或 `--参数名 值` 传入，未传时使用
函数默认值。与脚本选项重名的标量可写成 `--arg repeat=10`。

在 `h20` 上使用 `learn-cuda` 环境：

```bash
conda activate learn-cuda
cd /root/gaoxinlong/Desktop/repo/learn-leetgpu
python benchmark_triton.py tests/triton/smoke_vector_add.py --N=1000000
# 非交互 SSH 未激活 conda 时，也可直接使用：
# /root/miniforge3/envs/learn-cuda/bin/python benchmark_triton.py ...
```

矩阵乘法 / GEMM，`a, b, c, M, N, K, alpha, beta` 签名：

```bash
python benchmark_triton.py /path/to/gemm.py \
  --M=512 --N=768 --K=256 --alpha=1.0 --beta=0.0 \
  --shape a=MxK --shape b=KxN --shape c=MxN \
  --dtype=float32 --gpu=0 --warmup=10 --repeat=100
```

二维卷积，`input, kernel, output, input_rows, input_cols, kernel_rows,
kernel_cols` 签名。下面的 output shape 适用于 stride=1、无 padding 的 valid
卷积；其他语义请按实际 kernel 修改：

```bash
python benchmark_triton.py /path/to/convolution.py \
  --input_rows=128 --input_cols=256 --kernel_rows=3 --kernel_cols=3 \
  --shape input=128x256 --shape kernel=3x3 --shape output=126x254 \
  --warmup=10 --repeat=100 --json=conv_benchmark.json
```

- `--shape NAME=...` 可重复使用，维度用 `x` 或逗号分隔，可引用 int 参数名。
  所有 Tensor 均为连续布局；签名本身不能确定真实 shape、dtype 或输入输出角色。
- 不指定 shape 时，按 `benchmark.py` 的规则为每个 Tensor 分配一维缓冲区：取
  正整数参数中最大的两个相乘；只有一个则用它；没有则分配 1048576 个元素。
  这是容量估算，不能保证适合任意 kernel。与旧脚本不同，不会静默截断过大的容量。
- `--ptr-size=1048576` 覆盖未指定 shape 的 Tensor 元素数，单位是元素而非字节；
  显式 `--shape` 优先。`--dtype` 默认 float32，可用 `--tensor-dtype indices=int32`
  单独指定某个 Tensor 的类型。
- 浮点 Tensor 使用随机正态值，整数为 `[0, 8)`，bool 为 False。需要合法索引、
  稀疏结构或特殊数据分布的算子，应按其契约修改输入构造。
- 首次调用（含 Triton JIT / autotune）与 warmup 均不计时；统计 average、median、
  min、max，单位为 ms，同时展示 us。`--json` 保存每次耗时和完整参数配置。
- 计时范围是当前 PyTorch CUDA stream 上的 `solve`，可能包含 Python 发射导致的
  GPU 空隙。额外 stream 的工作必须在返回前汇合到当前 stream。所有调用复用相同
  buffer，原地更新会保留；此工具不测量冷缓存或每次重置输入的场景。
- 默认展示调用前后各 Tensor 的前 8 个元素，`--preview=0` 可关闭。这不是正确性
  校验，也不验证 `solve` 是否真正计算了预期结果。

该机器 `learn-cuda` 环境为 Python 3.12.14、PyTorch 2.11.0+cu126、
Triton 3.6.0。测试代码在 `tests/triton/`，包括三类真实 Triton kernel、
参数解析测试，以及与 CPU PyTorch 对照的数值测试。测试 kernel 只用于
验证 harness，不是优化后的性能实现。数值测试使用 GPU 0。

```bash
python tests/triton/test_harness.py
python tests/triton/check_numerics.py
python benchmark_triton.py --help
```
