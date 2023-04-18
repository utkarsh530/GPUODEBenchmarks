"""
Benchmarking of the Julia's EnsembleGPUArray GPU acceleration. The implementation is similar
to the vectorized map approach. The timings are stored in ./data folder, with Julia_EnGPUArray_times
".txt" file.

Created by: Utkarsh
Last Updated: 18 April 2023
"""


using DiffEqGPU, BenchmarkTools, StaticArrays, OrdinaryDiffEq
using CUDA

@show ARGS
#settings

numberOfParameters = isinteractive() ? 1024 : parse(Int64, ARGS[1])
gpuID = 0

device!(CuDevice(gpuID))
println("Running on " * string(CuDevice(gpuID)))

function lorenz(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - 2.666f0 * u[3]
    return @SVector [du1, du2, du3]
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 1.0f0)
p = @SArray [21.0f0]
prob = ODEProblem(lorenz, u0, tspan, p)

## parameter list uniformly varying the single lorenz parameter
parameterList = range(0.0f0, stop = 21.0f0, length = numberOfParameters)

lorenzProblem = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(lorenzProblem, prob_func = prob_func)

batch = 1:numberOfParameters
if ensembleProb.safetycopy
    probs = map(batch) do i
        ensembleProb.prob_func(deepcopy(ensembleProb.prob), i, 1)
    end
else
    probs = map(batch) do i
        ensembleProb.prob_func(ensembleProb.prob, i, 1)
    end
end

@info "Solving the problem"
data = @benchmark @CUDA.sync DiffEqGPU.vectorized_map_solve($probs, Tsit5(),EnsembleGPUArray(0.0), $batch, false,dt = 0.001f0, save_everystep = false, dense = false)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "Julia_EnGPUArray_times_unadaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

data = @benchmark @CUDA.sync DiffEqGPU.vmap_solve($probs, Tsit5(),EnsembleGPUArray(0.0), $I, true,dt = 0.001f0, save_everystep = false, dense = false)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "Julia_EnGPUArray_times_adaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
