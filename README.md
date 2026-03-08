# 命令

benckmark 命令：

```
benchmark.py easy/Matrix\ Multiplication/solution.cu --M=8192 --K=6144 --N=4096 --repeat=20
```

- 自动编译 cu 文件
- 自动捕获参数名后创建对应大小的 Tensor, M、K、N 等随 kernel extern 包装函数变化
- 不测试正确性, 只测试调用性能

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
        -o profile_log -f \
        $(which python) benchmark.py easy/Matrix\ Multiplication/solution_register_prefetch.cu --M=8192 --K=6144 --N=4096 --repeat=20
```
