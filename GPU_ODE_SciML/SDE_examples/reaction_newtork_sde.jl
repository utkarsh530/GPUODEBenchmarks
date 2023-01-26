
using Catalyst, OrdinaryDiffEq, Plots, StochasticDiffEq, StaticArrays, CUDA

### Example 4: Ensemble SDE simulations (varioous parameter values) at steady state behaviours of 4 variable CRN (Generalised bacterial stress response model). ###

# Declare the model (using Catalys).
σGen_system = @reaction_network begin
    (v0 + (S * σ)^n / ((S * σ)^n + (D * A3)^n + 1), 1.0), ∅ ↔ σ
    (σ / τ, 1 / τ), ∅ ↔ A1
    (A1 / τ, 1 / τ), ∅ ↔ A2
    (A2 / τ, 1 / τ), ∅ ↔ A3
end S D τ v0 n η

# Declares the parameter values.
σGen_parameters = [:S => 2.3, :D => 5.0, :τ => 10.0, :v0 => 0.1, :n => 3, :η => 0.1]

# Set ensemble parameter values.
trajectories = 100
S_vals = LinRange(0.2, 20.0, trajectories)
function σGen_p_func(prob, i, repeat)
    # prob_new = deepcopy(prob)
    # prob_new.p[1] = S_vals[i]
    # return prob_new
    remake(prob; p = [S_vals[i], prob.p[2:end]...])
end

# Declare initial condition.
σGen_u0 = [:σ => 0.1, :A1 => 0.1, :A2 => 0.1, :A3 => 0.1] # (for some S values, the system will start far away from the steady state).

# Create EnsembleProblem.
σGen_sprob = SDEProblem(σGen_system, σGen_u0, (0.0, 1000.0), σGen_parameters,
                        noise_scaling = (@parameters η)[1])

### Experimentation
sys = modelingtoolkitize(σGen_sprob)
T1 = Float32
prob = SDEProblem{false}(sys, SVector{length(σGen_sprob.u0), T1}(σGen_sprob.u0),
                         Float32.(σGen_sprob.tspan),
                         SVector{length(σGen_sprob.p), T1}(σGen_sprob.p),
                         noise_rate_prototype = SMatrix{
                                                        size(σGen_sprob.noise_rate_prototype)...,
                                                        T1}(σGen_sprob.noise_rate_prototype))

using DiffEqGPU

prob_func = (prob, i, repeat) -> remake(prob,
                                        p = SVector{6, T1}(S_vals[i], prob.p[2:end]...))
eprob = EnsembleProblem(prob, prob_func = prob_func)

saveat = T1(0.0f0):T1(1.0f0):T1(1000.0f0)
dt = T1(1.0f0)
## Check if solve works on GPU
sol = solve(eprob, GPUEM(), EnsembleGPUKernel(0.0); dt, adaptive = false, trajectories,
            saveat)

## Uncomment to Plot
# using Plots
# plot(sol.u[5], idxs = 1, label = "S = $(S_vals[5])")
# plot!(sol.u[11], idxs = 1, label = "S = $(S_vals[10])")
# plot!(sol.u[20], idxs = 1, label = "S = $(S_vals[15])")
# plot!(sol.u[45], idxs = 1, label = "S = $(S_vals[45])")
# plot!(sol.u[75], idxs = 1, label = "S = $(S_vals[75])")

### Benchmarking
using BenchmarkTools

### Without saveat

@benchmark solve($eprob, GPUEM(), EnsembleGPUKernel(0.0); dt, adaptive = false,
                 trajectories, save_everystep = false)

@benchmark solve($eprob, EM(), EnsembleThreads(); dt, adaptive = false, trajectories,
                 save_everystep = false)

### Lower lower API

probs = map(1:trajectories) do i
    prob_func(prob, i, false)
end;

## Move the arrays to the GPU
probs = cu(probs);

## Finally use the lower API for faster solves! (Fixed time-stepping)

# Benchmarking
@benchmark solve($eprob, GPUEM(), EnsembleGPUKernel(0.0); dt = 1.0f0, adaptive = false,
                 trajectories, save_everystep = false)

@benchmark CUDA.@sync ts, us = DiffEqGPU.vectorized_solve($probs, $prob, GPUEM();
                                                          save_everystep = false,
                                                          dt = 1.0f0)

@benchmark sol = solve($eprob, EM(), EnsembleThreads(); dt = 1.0f0, adaptive = false,
                       trajectories, save_everystep = false)
