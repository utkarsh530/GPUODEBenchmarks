
using DiffEqGPU, BenchmarkTools, StaticArrays, OrdinaryDiffEq


@show ARGS
#settings

include(joinpath(@__DIR__,"./problems.jl"))

numberOfParameters = isinteractive() ? 2097152 : parse(Int64, ARGS[1])

parameterList = range(0.0f0, stop = 1f4, length = numberOfParameters)

prob_func = (prob, i, repeat) -> remake(rober_prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(rober_prob, prob_func = prob_func)

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

## Make them compatible with Backend
probs = if ARGS[2] == "CUDA"
    using CUDA
    cu(probs)
elseif ARGS[2] == "oneAPI"
    using oneAPI
    probs |> oneArray
elseif ARGS[2] == "AMDGPU"
    using AMDGPU
    roc(probs)
elseif ARGS[2] == "Metal"
    using Metal
    probs |> MtlArray
end

@info "Solving the problem"
# data = if ARGS[2] == "CUDA"
#     @benchmark CUDA.@sync DiffEqGPU.vectorized_solve($probs, $ensembleProb.prob,
#                                                      GPURosenbrock23();
#                                                      save_everystep = false,
#                                                      dt = 0.001f0)
# elseif ARGS[2] == "oneAPI"
#     @benchmark oneAPI.@sync DiffEqGPU.vectorized_solve($probs, $ensembleProb.prob,
#                                                      GPURosenbrock23();
#                                                        save_everystep = false,
#                                                        dt = 0.001f0)
# elseif ARGS[2] == "AMDGPU"
#     @benchmark DiffEqGPU.vectorized_solve($probs, $ensembleProb.prob,
#                                           GPURosenbrock23();
#                                           save_everystep = false,
#                                           dt = 0.001f0)
# elseif ARGS[2] == "Metal"
#     @benchmark Metal.@sync DiffEqGPU.vectorized_solve($probs, $ensembleProb.prob,
#                                                       GPURosenbrock23();
#                                                       save_everystep = false,
#                                                       dt = 0.001f0)
# end

# if !isinteractive()
#     open(joinpath(dirname(@__DIR__), "data", "devices", ARGS[2],
#                   "Julia_times_unadaptive.txt"), "a+") do io
#         println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
#     end
# end

# println("Parameter number: " * string(numberOfParameters))
# println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
# println("Allocs: " * string(data.allocs))

data = if ARGS[2] == "CUDA"
    @benchmark CUDA.@sync DiffEqGPU.vectorized_asolve($probs, $ensembleProb.prob,
                                                      GPURosenbrock23();
                                                      dt = 0.001f0)

elseif ARGS[2] == "oneAPI"
    @benchmark oneAPI.@sync DiffEqGPU.vectorized_asolve($probs, $ensembleProb.prob,
                                                        GPURosenbrock23();
                                                        dt = 0.001f0)

elseif ARGS[2] == "AMDGPU"
    @benchmark DiffEqGPU.vectorized_asolve($probs, $ensembleProb.prob,
                                           GPURosenbrock23();
                                           dt = 0.001f0)

elseif ARGS[2] == "Metal"
    @benchmark Metal.@sync DiffEqGPU.vectorized_asolve($probs, $ensembleProb.prob,
                                                       GPURosenbrock23();
                                                       dt = 0.001f0)
end

if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "devices", ARGS[2],
                  "Julia_times_adaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
