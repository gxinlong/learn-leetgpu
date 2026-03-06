/**
 * 仅供 IDE/clangd 解析 .cu 时使用，消除 threadIdx/blockIdx/blockDim 未定义报错。
 * 用 nvcc 编译时 __CUDACC__ 已定义，不会包含本段；这些变量由 CUDA 编译器在设备代码中自动提供。
 */
#ifndef __CUDACC__
struct __dim3_ide_stub {
  unsigned int x, y, z;
};
extern __dim3_ide_stub threadIdx, blockIdx, blockDim;
extern __dim3_ide_stub gridDim;
#endif
