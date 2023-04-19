# GPUODEBenchmarks
Comparsion of Julia's GPU based ensemble ODE solvers with other open-source implementations in C++, JAX and PyTorch. These artifacts are part of the paper:
> Automated Translation and Accelerated Solving of Differential Equations on Multiple GPU Platforms

**_NOTE:_**  This repository is meant to contain scripts for benchmarking existing ensemble ODE solvers. For external purposes, one can directly use the solvers from the respective libraries. 

### Performance comparsion with other open-source ensemble ODE solvers
<img src="https://github.com/utkarsh530/GPUODEBenchmarks/blob/main/paper_artifacts/figures/Lorenz_unadaptive.png" alt="drawing" width="50%"/>

### Works with NVIDIA, Intel, AMD and Apple GPUs
<img src="https://github.com/utkarsh530/GPUODEBenchmarks/blob/main/paper_artifacts/figures/Multi_GPU_unadaptive.png" alt="drawing" width="50%"/>

# Reproduction of the benchmarks
Running the benchmarks requires setting up packages for different programs. Please follow `README_<program>.md` for setting up and installation instructions. After the setup and installation is done, the timings for the ODE solvers can be simply done via the bash script `run_benchmark.sh`. The syntax is:

```console
$ ./run_benchmark.sh -p <program> -d <device> -m <model> -n <max_trajectories>
```

With the acceptable flag arguments as:

```
<program> = {"julia","jax","pytorch","cpp"}
<device> = {"cpu","gpu"}
<model> = {"ode","sde"}
<max_trajectories> = N (Eg. 1024)
```
The script will run the respective program for different trajectories $n$ for $8\le n \le N$, with jumps of multiple of 4. The scripts for JAX, PyTorch and C++ (MPGOS) is available only for GPU ODE solvers.

## Configuring GPU

For the purpose of benchmarking the ODE solvers, the NVIDIA CUDA backend is used. Please ensure that all related drivers and CUDA Toolkit is installed in your workstation. The recommended CUDAToolkit is >= 11. One can check the installation by runnning:

```console
$ nvidia-smi
```
If the toolkit is installed correctly, one will get a message similar to below:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.108.03   Driver Version: 510.108.03   CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:D8:00.0 Off |                  Off |
| N/A   28C    P0    25W / 150W |      0MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
...
```
Additionally, for benchmarking CUDA C++, the NVIDIA CUDA C++ compiler is also required. One can check for the installation as:

```console
$ nvcc
```
When installed successfully, should return something like this:

```
nvcc fatal   : No input files specified; use option --help for more information
```
**_NOTE:_**: For using CUDA with Julia, one can use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) directly, as described in the section "Getting started with Julia". It installs the requires binaries automatically.  


## Getting started with Julia

### Install Julia
Firstly, we'll need to install Julia. The user can download the binaries from the [official JuliaLang website](https://julialang.org/downloads/) or follow this [tutorial](https://julialang.org/downloads/platform/). Alternatively, one can use the convenience of a [Julia multiplier](https://github.com/JuliaLang/juliaup). The recommended OS for installation is Linux. **The recommended Julia version is v1.8**. For using AMD GPUs, please install v1.9.

### Add Julia to your PATH
Execute this command in your shell or add this entry to your `.bashrc` or `.profile` file:

```console
$ export PATH="$PATH:/path/to/<Julia directory>/bin"
````

Now try:

```console
$ julia
```

If the steps are followed correctly, the Julia terminal will show without any errors.

## Getting ready for running Julia GPU solvers

### Installing CUDA.jl

We'll need to install CUDA.jl for benchmarking. It is the only backend which is compatible with the ODE solvers in JAX, PyTorch and MPGOS. However, our ODE solvers are compatible with multiple backends. See details further for running the ODE solvers with different backends.
To do so, one can simply follow the below process in the Julia Terminal:

```julia
using Pkg
Pkg.activate()
Pkg.update()
Pkg.install("CUDA")
```
This installs the CUDA library in the global enviroment. This might take some while. After connection, try running:

```julia
using CUDA
CUDA.versioninfo()
CuArray(rand(2)) #Testing by allocating an array on GPU
```
If the above steps runs without errors, congratulations, your CUDA.jl installation is successful.

### Instantiating libraries for benchmarking

The GPU solvers are part of the repository, DiffEqGPU.jl. These scripts simply invoke the solvers from the library and collect timings for benchmarking. We will instantiate and precompile all the packages beforehand to avoid the wait times during benchmarking. The folder `./GPU_ODE_Julia` contains all the related scripts for the GPU solvers. Start the Julia session as:

```console
$ julia --project="/path/to/<GPUODEBenchmarks>/GPU_ODE_Julia" --threads=auto
```

Now, in the Julia terminal, run:

```julia
using Pkg
Pkg.instantiate()
Pkg.precompile()
```

It might take some time to precompile.

### Running benchmarks

Now we are set to benchmark the ODE solvers. To do so, simply in the console, type:

```console
$ cd /path/to/<GPUODEBenchmarks>
$ bash ./run_benchmark.sh -p julia -d gpu -m ode
```
