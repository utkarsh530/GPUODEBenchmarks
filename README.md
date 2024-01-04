# GPUODEBenchmarks
Comparison of Julia's GPU-based ensemble ODE solvers with other open-source implementations in C++, JAX, and PyTorch. These artifacts are part of the paper:
> Automated Translation and Accelerated Solving of Differential Equations on Multiple GPU Platforms

**_NOTE:_**  This repository is meant to contain scripts for benchmarking existing ensemble ODE solvers. For external purposes, one can directly use the solvers from the respective libraries. 

### Performance comparison with other open-source ensemble ODE solvers
<img src="https://github.com/utkarsh530/GPUODEBenchmarks/blob/main/paper_artifacts/figures/Lorenz_unadaptive.png" alt="drawing" width="50%"/>

### Works with NVIDIA, Intel, AMD, and Apple GPUs
<img src="https://github.com/utkarsh530/GPUODEBenchmarks/blob/main/paper_artifacts/figures/Multi_GPU_unadaptive.png" alt="drawing" width="50%"/>

# Reproduction of the benchmarks

The methods are written in Julia and are part of the repository
<https://github.com/SciML/DiffEqGPU.jl>. The benchmark suite also
consists of the raw data, such as simulation times and plots mentioned
in the paper. The supported OS for the benchmark suite is Linux.

## Installing Julia

Firstly, we will need to install Julia. The user can download the
binaries from the official JuliaLang website
[`https://julialang.org/downloads/`](https://julialang.org/downloads/).
Alternatively, one can use the convenience of a Julia version
multiplexer, <https://github.com/JuliaLang/juliaup>. The recommended OS
for installation is Linux. The recommended Julia installation version is
v1.8. To use AMD GPUs, please install v1.9. The Julia installation
should also be added to the user's path.

## Setting up DiffEqGPU.jl

### Installing backends

The user must install the GPU backend library for testing
DiffEqGPU.jl-related code.

```julia
    julia> using Pkg
    julia> #Run either of them
    julia> Pkg.add("CUDA") # NVIDIA GPUs
    julia> Pkg.add("AMDGPU") #AMD GPUs
    julia> Pkg.add("oneAPI") #Intel GPUs
    julia> Pkg.add("Metal") #Apple M series GPUs
```
### Testing DiffEqGPU.jl

DiffEqGPU.jl is a test suite that regularly checks functionality by
testing features like multiple backend support, event handling, and
automatic differentiation. To test the functionality, one can follow the
below instructions. The user needs to specify the \"backend\" for
example \"CUDA\" for NVIDIA, \"AMDGPU\" for AMD, \"oneAPI\" for Intel
, and \"Metal\" for Apple GPUs. The estimated time of completion is 20
minutes.
```julia
    $ julia --project=.
    julia> using Pkg
    julia> Pkg.instantiate()
    julia> Pkg.precompile()
```
Finally, test the package with this command
```bash
    $ backend="CUDA"
    $ julia --project=. test_DiffEqGPU.jl $backend
```
Additionally, the GitHub discussion
[`https://github.com/SciML/DiffEqGPU.jl/issues/224#issuecomment-1453769679`](https://github.com/SciML/DiffEqGPU.jl/issues/224#issuecomment-1453769679)
highlights the use of textured memory with ODE solvers, accelerates the
code by $2\times$ over CPU.

### Continuous Integration and Development

DiffEqGPU.jl is a fully featured library with regression testing, semver
versioning, and version control. The tests are performed on cloud
machines having a multitude of different GPUs
[`https://buildkite.com/julialang/diffeqgpu-dot-jl/builds/705`](https://buildkite.com/julialang/diffeqgpu-dot-jl/builds/705).
These tests are approximately complete in 30 minutes. The publicly visible
testing framework serves as a testimonial of compatibility with multiple
platforms and said features in the paper.

## Testing GPU-accelerated ODE Benchmarks with other programs

### Benchmarking Julia (DiffEqGPU.jl) methods
We will need to install CUDA.jl for benchmarking. It is the only backend
compatible with the ODE solvers in JAX, PyTorch, and MPGOS. To do so,
one can follow the below process in the Julia Terminal:
```julia
    $ julia
    julia> using Pkg
    julia> Pkg.add("CUDA")
```
Let's clone the benchmark suite repository to start benchmarking;
```bash
    $ git clone https://github.com/utkarsh530\
    /GPUODEBenchmarks.git
```
We will instantiate and pre-compile all the packages beforehand to avoid
the wait times during benchmarking. The folder ./GPU_ODE_Julia contains
all the related scripts for the GPU solvers.
```bash
    $ cd ./GPUODEBenchmarks
    $ julia --project=./GPU_ODE_Julia --threads=auto
    julia> using Pkg
    julia> Pkg.instantiate()
    julia> Pkg.precompile()
    julia> exit()
```
It may take a few minutes to complete (\< 10 minutes). After this, we
can generate the timings of ODE solvers written in Julia. There is a
script to benchmark ODE solvers for the different number of trajectories
to demonstrate scalability and performance. The script invocation and
timings can be generated through the following:
```bash
    $ bash ./run_benchmark.sh -l julia -d gpu -m ode
```
It might take around 20 minutes to finish. The flag `-n N` can be used
to specify the upper bound of the trajectories to benchmark. By default
$N = 2^{24}$, where the simulation runs for $n \in 8 \le n < N$, with
the multiples of $4$.

The data will be generated in the `data/Julia` directory, with two files
for fixed and adaptive time-stepping simulations. The first column in
the \".txt\" file will be the number of trajectories, and the section
column will contain the time in milliseconds.

Additionally, to benchmark ODE solvers for other backends:
```bash
    $ N = $((2**24))
    Benchmark
    $ backend = "Metal"
    $ ./runner_scripts/gpu/run_ode_mult_device.sh\
    $N $backend
```
### Benchmarking C++ (MPGOS) ODE solvers

Benchmarking MPGOS ODE solvers requires the CUDA C++ compiler to be
installed correctly. The recommended CUDA Toolkit version is \>= 11. The
installation can be checked through:
```bash
    $ nvcc
    If the installation exists, it will return 
    something like this:
    nvcc fatal   : No input files specified; 
    use option --help for more information
```
If `nvcc` is not found, the user must install the CUDA Toolkit. The
NVIDIA's website lists the resource
[`https://developer.nvidia.com/cuda-downloads`](https://developer.nvidia.com/cuda-downloads)
for installation.

The MPGOS scripts are in the `GPU_ODE_MPGOS` folder. The file
`GPU_ODE_MPGOS/Lorenz.cu` is the main executed code. However, the MPGOS
programs can be run with the same bash script by changing the arguments
as:
```bash
    $ bash ./run_benchmark.sh -l cpp -d gpu -m ode
```
It will generate the data files in the `data/cpp` folder.

### Benchmarking JAX (Diffrax) ODE solvers

Benchmarking JAX-based ODE solvers require installing Python 3.9 and
`conda`. First, we will install all the Python packages for
benchmarking:
```bash
    $ conda env create -f environment.yml
    $ conda activate venv_jax
```
It should install the correct version of JAX with CUDA enabled and the
Diffrax library. The GitHub
[`https://github.com/google/jax#installation`](https://github.com/google/jax#installation)
is a guide to follow if the installation fails.

For our purposes, we can benchmark the solvers by:
```bash
    $ bash ./run_benchmark.sh -l jax -d gpu -m ode
```

#### A note on JIT ordering in JAX

The JIT ordering JAX matters and sometimes can enhance performance if done correctly. We have tested that vmap and JIT ordering does not make a noticeable difference in our case. The results are available at this [Colab notebook](https://colab.research.google.com/drive/1d7G-O5JX31lHbg7jTzzozbo5-Gp7DBEv?usp=sharing).

### Benchmarking PyTorch (torchdiffeq) ODE solvers

Benchmarking PyTorch-based ODE solvers is a similar process compared to
JAX ones.
```bash
    $ conda env create -f environment.yml
    $ conda activate venv_torch
```
`torchdiffeq` does not fully support vectorized maps with ODE solvers.
To circumvent this, we extended the functionality by rewriting some
library parts. To download it:
```bash
    (venv_torch)$ pip uninstall torchdiffeq
    (venv_torch)$ pip uninstall torchdiffeq
    (venv_torch)$ pip install git+https://github.com/\
    utkarsh530/torchdiffeq.git@u/vmap
```
Then run the benchmarks by:
```bash
    $ bash ./run_benchmark.sh -l pytorch -d gpu -m ode
```
## Comparing GPU acceleration of ODEs with CPUs

The benchmark suite can also be used to test the GPU acceleration of ODE
solvers in comparison with CPUs. The process for generating simulation
times for GPUs can be done by following the GPU section mentioned earlier. The following bash script
allows the generation of CPU simulation times for ODEs:
```bash
    $ bash ./run_benchmark.sh -l julia -d cpu -m ode
```
The simulation times will be generated in `data/CPU`. Each of the
workflow takes approximately 20 minutes to finish.

## Benchmarking GPU acceleration of SDEs with CPUs

The SDE solvers in Julia are benchmarked by comparing them to the
CPU-accelerated simulation. This will benchmark the linear SDE with
three states, as described in the \"Benchmarks and case studies\"
section. To generate simulation times for GPU, do the following:
```bash
    $ bash ./run_benchmark.sh -l julia -d gpu -m sde
```
We can generate the simulation times for CPU-accelerated codes through the following:
```bash
    $ bash ./run_benchmark.sh -l julia -d cpu -m sde
```
The results will get generated in `data/SDE` and `data/CPU/SDE`, taking
around 10 minutes to complete.

## Composability with MPI

Julia supports Message Passing Interface (MPI) to allow Single Program
Multiple Data (SPMD) type parallel programming. The composability of the
GPU ODE solvers enable seamless integration with MPI, enabling scaling
the ODE solvers to clusters on multiple nodes.
```julia
    $ julia --project=./GPU_ODE_Julia
    julia> using Pkg
    # install MPI.jl
    julia> Pkg.add("MPI")
```
An example script solving the Lorenz problem for approximately 1 billion
parameters are available in the `MPI` folder. A SLURM-based script is
shown below.
```bash
    #!/bin/bash
    # Slurm Sbatch Options
    # Reqeust no. of GPUs/node
    #SBATCH --gres=gpu:volta:1
    # 1 process per node 
    #SBATCH -n 5 -N 5
    #SBATCH --output="./mpi_scatter_test.log-%j"
    # Loading the required module

    # MPI.jl requires a memory pool to be disabled
    export JULIA_CUDA_MEMORY_POOL=none
    export JULIA_MPI_BINARY=system
    # Use local CUDA toolkit installation
    export JULIA_CUDA_USE_BINARYBUILDER=false

    source $HOME/.bashrc
    module load cuda mpi

    srun hostname > hostfile
    time mpiexec julia --project=./GPU_ODE_Julia\ 
    ./MPI/gpu_ode_mpi.jl
```
## Plotting Results

The plotting scripts to visualize the simulation times. The scripts are
located in the `runner_scripts/plot` folder. These scripts replicate the
benchmark figures in the paper. The benchmark suite contains the
simulation data generated by authors, which can be used to verify the
plots. Various benchmarks can be plotted, which are described in the
different sections. The plotting scripts are based on Julia. As a
preliminary step:
```julia
    $ cd GPUODEBenchmarks
    $ julia project=.
    julia> using Pkg
    julia> Pkg.instantiate()
    julia> Pkg.precompile()
```
The plot comparison between Julia, C++, JAX, and PyTorch mentioned in
the paper can be generated by using the below command:
```bash
    $ julia --project=. ./runner_scripts/plot\
    /plot_ode_comp.jl
```
The plot will get saved in the `plots` folder.

Similarly, the other plots in the paper can be generated by running the
different scripts in the folder `runner_scripts/plot`.
```bash
    plot performance of GPU ODE solvers 
    with multiple backends
    $ julia --project=. ./runner_scripts/plot\
    /plot_mult_gpu.jl 
    plot GPU ODE solvers comparsion with CPUs
    $ julia --project=. ./runner_scripts/plot\
    /plot_ode_comp.jl 
    plot GPU SDE solvers comparsion with CPUs
    $ julia --project=. ./runner_scripts/plot\
    /plot_sde_comp.jl 
    plot CRN Network sim comparison with CPUs
    $ julia --project=. ./runner_scripts/plot\
    /plot_sde_crn.jl 
```
To plot data generated by running the scripts, specify the location of
the `data` as the argument to the mentioned command.
```bash
    $ julia --project=. ./runner_scripts/plot/\
    plot_mult_gpu.jl /path/to/data/
```
