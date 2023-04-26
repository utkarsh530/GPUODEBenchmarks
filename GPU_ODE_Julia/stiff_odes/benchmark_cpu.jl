using DiffEqGPU, BenchmarkTools, StaticArrays, OrdinaryDiffEq

using Suppressor
@show ARGS
#settings

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

function rober_f(internal_var___u, internal_var___p, t)
    internal_var___du1 = -(0.04e0) * internal_var___u[1] +
                         internal_var___p[1] * internal_var___u[2] *
                         internal_var___u[3]
    internal_var___du2 = (0.04e0 * internal_var___u[1] -
                          3.0e7 * internal_var___u[2]^2) -
                         internal_var___p[1] * internal_var___u[2] *
                         internal_var___u[3]
    internal_var___du3 = 3.0e7 * internal_var___u[2]^2
    return SVector{3,eltype(internal_var___u)}(internal_var___du1, internal_var___du2, internal_var___du3)
end

function rober_jac(internal_var___u, internal_var___p, t)
    internal_var___J11 = -(0.04e0)
    internal_var___J12 = internal_var___p[1] * internal_var___u[3]
    internal_var___J13 = internal_var___p[1] * internal_var___u[2]
    internal_var___J21 = 0.04e0 * 1
    internal_var___J22 = -2 * 3.0e7 * internal_var___u[2] -
                         internal_var___p[1] * internal_var___u[3]
    internal_var___J23 = -(internal_var___p[1]) * internal_var___u[2]
    internal_var___J31 = 0 * 1
    internal_var___J32 = 2 * 3.0e7 * internal_var___u[2]
    internal_var___J33 = 0 * 1
    return SMatrix{3, 3, eltype(internal_var___u)}(internal_var___J11, internal_var___J21, internal_var___J31,
                         internal_var___J12, internal_var___J22, internal_var___J32,
                         internal_var___J13, internal_var___J23, internal_var___J33)
end

function rober_tgrad(u, p, t)
    return SVector{3, eltype(u)}(0.0, 0.0, 0.0)
end

u0 = @SVector Float64[1.0, 0.0, 0.0]
p = @SVector Float64[1.0e4]

rober_prob = ODEProblem(ODEFunction(rober_f, jac = rober_jac, tgrad = rober_tgrad),
                        u0, (0.0e0, 1.0e5), p)

parameterList = range(0.0e0, stop = 1e4, length = numberOfParameters)

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

## Make them compatible with CUDA

@info "Solving the problem"

# DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob,
#                                                          GPURosenbrock23();
#                                                          dt = 0.001f0)

data = @suppress begin
    @benchmark DiffEqGPU.vectorized_asolve($probs, $ensembleProb.prob,
                                                            GPURosenbrock23();
                                                            dt = 0.001)
end

# data = @benchmark solve($ensembleProb, Rosenbrock23(), EnsembleThreads(), dt = 0.001,
#                         adaptive = true, save_everystep = false,
#                         trajectories = numberOfParameters)


if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "CPU", "stiff", "Julia_times_adaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
