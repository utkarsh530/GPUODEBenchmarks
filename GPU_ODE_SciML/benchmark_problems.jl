using GPU_ODE_SciML
using DiffEqGPU, BenchmarkTools, StaticArrays, SimpleDiffEq, OrdinaryDiffEq
using CUDA
using BenchmarkTools

@show ARGS
#settings

numberOfParameters = isinteractive() ? 1024 : parse(Int64, ARGS[1])
gpuID = 0

@show device!(CuDevice(gpuID))

problems_path = joinpath(@__DIR__, "problems")

include(joinpath(problems_path, "pleaides.jl"))

# Adaptive = false

numberOfParameters_array = [8 * 4^i for i in 0:3]

data_times = []
for numberOfParameters in numberOfParameters_array
    @info numberOfParameters
    data = @benchmark solve($ensembleProb, Tsit5(), EnsembleThreads(),
                            trajectories = numberOfParameters,
                            adaptive = false, save_everystep = false, dt = dt)

    time_cpu_threaded = minimum(data.times) / 1e6
    # b2 = @btime solve($monteprob, Tsit5(), EnsembleCPUArray(),
    #                   trajectories = numberOfParameters, adaptive = false, dt = dt)

    data = @benchmark CUDA.@sync solve($ensembleProb, GPUTsit5(), EnsembleGPUKernel(),
                                       trajectories = numberOfParameters, adaptive = false,
                                       save_everystep = false,
                                       dt = dt)

    time_gpu_kernel = minimum(data.times) / 1e3

    data = @benchmark CUDA.@sync DiffEqGPU.vectorized_solve(probs, ensembleProb.prob,
                                                            GPUTsit5();
                                                            save_everystep = false, dt = dt)

    time_gpu_kernel_lower = minimum(data.times) / 1e3

    push!(data_times, [time_cpu_threaded, time_gpu_kernel, time_gpu_kernel_lower])
end
