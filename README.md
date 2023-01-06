# GPUKernelBenchmarks
Comparsion of Julia's GPU Kernel based ODE solvers with other open-source GPU ODE solvers
## Benchmark information

```
julia> CUDA.versioninfo()
CUDA toolkit 11.6, local installation
NVIDIA driver 510.47.3, for CUDA 11.6
CUDA driver 11.6

Libraries:
- CUBLAS: 11.8.1
- CURAND: 10.2.9
- CUFFT: 10.7.0
- CUSOLVER: 11.3.2
- CUSPARSE: 11.7.1
- CUPTI: 16.0.0
- NVML: 11.0.0+510.47.3
- CUDNN: 8.30.2 (for CUDA 11.5.0)
- CUTENSOR: missing

Toolchain:
- Julia: 1.8.1
- LLVM: 13.0.1
- PTX ISA support: 3.2, 4.0, 4.1, 4.2, 4.3, 5.0, 6.0, 6.1, 6.3, 6.4, 6.5, 7.0, 7.1, 7.2
- Device capability support: sm_35, sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80, sm_86

Environment:
- JULIA_CUDA_USE_BINARYBUILDER: false

1 device:
  0: Tesla V100-PCIE-32GB (sm_70, 31.745 GiB / 32.000 GiB available)
```
