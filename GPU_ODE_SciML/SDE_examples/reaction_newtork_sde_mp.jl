
using Catalyst, Plots, StochasticDiffEq, StaticArrays, CUDA

@show ARGS
#settings

N = isinteractive() ? 10 : parse(Int64, ARGS[1])

### Example 4: Ensemble SDE simulations (varioous parameter values) at steady state behaviours of 4 variable CRN (Generalised bacterial stress response model). ###

## Credits: Torkel Loman

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
S_grid = Float32.(10 .^ (range(-1.0, stop = 2, length = N)))
D_grid = Float32.(10 .^ (range(-1, stop = 2, length = N)))
τ_grid = Float32[0.1, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.50, 10.0,
                 15.0, 20.0, 30.0, 50.0, 75.0, 100.0][1:2:19]
v0_grid = Float32[0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.20]
n_grid = Float32[2.0, 3.0, 4.0]
η_grid = Float32[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

parameters = collect(Iterators.product(S_grid, D_grid, τ_grid, v0_grid, n_grid, η_grid));

numberOfParameters = length(parameters)

@show numberOfParameters

function σGen_p_func(prob, i, repeat)
    for parameter in parameters
        remake(prob; p = SVector{6, T1}(parameter))
    end
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

function prob_func(prob, i, repeat)
    remake(prob; p = SVector{6, T1}(parameters[i]...))
end

eprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)

saveat = T1(0.0f0):T1(1.0f0):T1(1000.0f0)
dt = T1(0.1f0)

probs = map(1:numberOfParameters) do i
    prob_func(prob, i, false)
end;

### Benchmarking
using BenchmarkTools

@info "Solving the problem: GPU"

## Move the arrays to the GPU
gpuprobs = cu(probs);

## Finally use the lower API for faster solves! (Fixed time-stepping)

# Benchmarking

# CUDA.@time DiffEqGPU.vectorized_solve(gpuprobs, prob, GPUEM();
#                                             save_everystep = false,
#                                             dt = 0.1f0)

# @time DiffEqGPU.vectorized_solve(probs, prob, GPUEM();
#                                  save_everystep = false,
#                                  dt = 0.1f0)

data = @benchmark CUDA.@sync DiffEqGPU.vectorized_solve($gpuprobs, $prob, GPUEM();
                                                        save_everystep = false,
                                                        dt = 0.1f0)

if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "SDE", "MTK", "Julia_times_unadaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end
println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

@info "Solving the problem: CPU"

data = @benchmark DiffEqGPU.vectorized_solve($probs, $prob, GPUEM();
                                             save_everystep = false,
                                             dt = 0.1f0)

if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "CPU", "SDE", "MTK", "Julia_times_unadaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
