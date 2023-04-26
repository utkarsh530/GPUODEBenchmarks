using DiffEqGPU, BenchmarkTools, StaticArrays, OrdinaryDiffEq
using CUDA

@show ARGS
#settings

include(joinpath(@__DIR__,"./problems.jl"))

numberOfParameters = isinteractive() ? 8388608 : parse(Int64, ARGS[1])

parameterList = range(0.0f0, stop = 1f4, length = numberOfParameters)

prob_func = (prob, i, repeat) -> remake(rober_prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(rober_prob, prob_func = prob_func)

## Building problems here only
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

data = @benchmark CUDA.@sync DiffEqGPU.vectorized_map_solve($probs, Rosenbrock23(),
                                                            EnsembleGPUArray(0.0), $batch,
                                                            true, dt = 0.001f0,
                                                            save_everystep = false,
                                                            dense = false, dtmin = 1f-14)
if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "EnsembleGPUArray", "stiff",
                  "Julia_EnGPUArray_times_adaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
