using GPU_ODE_SciML
using DiffEqGPU, BenchmarkTools, StaticArrays, SimpleDiffEq
using CUDA

@show ARGS
#settings

numberOfParameters = isinteractive() ? 768000 : parse(Int64, ARGS[1])
gpuID = 0

device!(CuDevice(gpuID))
println("Running on " * string(CuDevice(gpuID)))

using Random
Random.seed!(123)

# 1D Linear ODE
function f(u::AbstractArray{T}, p, t::T) where {T}
    return 1.01f0 * u
end
function f_analytic(u₀, p, t)
    u₀ * exp(1.01 * t)
end

T = Float32

tspan = (0.0, 10.0)
tspan = T.(tspan)
u0 = @SVector rand(T, 100)
prob = ODEProblem(ODEFunction(f, analytic = f_analytic), u0, tspan)

ensembleProb = EnsembleProblem(prob)

sol = solve(ensembleProb, GPUTsit5(), EnsembleGPUKernel(), trajectories = 2, dt = 1.0f0)

### Lower level API ####

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
sol = @time CUDA.@sync DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob, GPUTsit5();
                                                   save_everystep = false, dt = 0.001f0)
