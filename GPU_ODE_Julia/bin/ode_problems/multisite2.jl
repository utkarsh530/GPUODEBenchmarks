using GPU_ODE_Julia
using DiffEqGPU, BenchmarkTools, StaticArrays, SimpleDiffEq, ReactionNetworkImporters,
      Catalyst
using CUDA

@show ARGS
#settings

numberOfParameters = isinteractive() ? 2 : parse(Int64, ARGS[1])
gpuID = 0

device!(CuDevice(gpuID))
println("Running on " * string(CuDevice(gpuID)))

prnbng = loadrxnetwork(BNGNetwork(), joinpath(@__DIR__, "Models/multisite2.net"))

rn = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

osys = convert(ODESystem, rn)

tf = 2.0
tspan = (0.0, tf)
oprob = ODEProblem{false}(osys, Float64[], tspan, Float64[])

T = Float64

prob = make_gpu_compatible(oprob, Val(T))

@assert prob.f(prob.u0, prob.p, T(1.0f0)) isa StaticArray{<:Tuple, T}

ensembleProb = EnsembleProblem(prob)

sol = solve(ensembleProb, GPUTsit5(), EnsembleGPUKernel(), trajectories = 2, dt = 0.001f0)

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
                                                   save_everystep = false, dt = T(0.001))
