
using DiffEqGPU, BenchmarkTools, StaticArrays, SimpleDiffEq

@show ARGS
#settings

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])


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

parameterList = range(0.0f0, stop = 21.0f0, length = numberOfParameters)

lorenzProblem = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(lorenzProblem, prob_func = prob_func)

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

# using oneAPI
# device()
# probs = probs |> oneArray

## Make them compatible with Backend
probs = if ARGS[2] == "CUDA"
    using CUDA
    gpuID = 0
    device!(CuDevice(gpuID))
    println("Running on " * string(CuDevice(gpuID)))
    cu(probs)
elseif ARGS[2] == "oneAPI"
    using oneAPI
    device()
    probs |> oneArray
elseif ARGS[2] == "AMDGPU"
    using AMDGPU
    roc(probs)
end

@info "Solving the problem"
data = @benchmark oneAPI.@sync DiffEqGPU.vectorized_solve($probs, $ensembleProb.prob,
                                                        GPUTsit5();
                                                        save_everystep = false,
                                                        dt = 0.001f0)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", ARGS[2], "Julia_times_unadaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

data = @benchmark oneAPI.@sync DiffEqGPU.vectorized_asolve($probs, $ensembleProb.prob,
                                                         GPUTsit5();
                                                         dt = 0.001f0, reltol = 1.0f-8,
                                                         abstol = 1.0f-8)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", ARGS[2], "Julia_times_adaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
