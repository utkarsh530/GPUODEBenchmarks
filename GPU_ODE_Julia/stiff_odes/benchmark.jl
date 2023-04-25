using DiffEqGPU, BenchmarkTools, StaticArrays, SimpleDiffEq
using CUDA

device!(1)

@show ARGS
#settings

include(joinpath(@__DIR__,"./problems.jl"))

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

ensembleProb = EnsembleProblem(rober_prob)

## Building problems here only
I = 1:numberOfParameters
if ensembleProb.safetycopy
    probs = map(I) do i
        ensembleProb.prob_func(deepcopy(ensembleProb.prob), i, 1)
    end
else
    probs = map(I) do i
        ensembleProb.prob_func(ensembleProb.prob, i, 1)
    end
end

## Make them compatible with CUDA
probs = cu(probs)

@info "Solving the problem"

# DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob,
#                                                          GPURosenbrock23();
#                                                          dt = 0.001f0)

data = @benchmark CUDA.@sync DiffEqGPU.vectorized_asolve($probs, $ensembleProb.prob,
                                                         GPURosenbrock23();
                                                         dt = 0.001f0)

if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "Julia", "stiff", "Julia_times_adaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
