# load packages
# using DiffEqFlux
# using SciMLSensitivity
# using Optimization
using StochasticDiffEq, DiffEqCallbacks, DiffEqNoiseProcess
using Statistics, LinearAlgebra, Random
using DiffEqGPU
using CUDA
# using Plots

#################################################
lr = 0.01f0
epochs = 100

numtraj = 16 # number of trajectories in parallel simulations for training
numtrajplot = 32 # .. for plotting

# time range for the solver
dt = 0.0005f0
tinterval = 0.05f0
tstart = 0.0f0
Nintervals = 20 # total number of intervals, total time = t_interval*Nintervals
tspan = (tstart, tinterval * Nintervals)
ts = Array(tstart:dt:(Nintervals * tinterval + dt)) # time array for noise grid

# Hamiltonian parameters
Δ = 20.0f0
Ωmax = 10.0f0 # control parameter (maximum amplitude)
κ = 0.3f0

# loss hyperparameters
C1 = Float32(1.0)  # evolution state fidelity

struct Parameters{flType, intType, tType}
    lr::flType
    epochs::intType
    numtraj::intType
    numtrajplot::intType
    dt::flType
    tinterval::flType
    tspan::tType
    Nintervals::intType
    ts::Vector{flType}
    Δ::flType
    Ωmax::flType
    κ::flType
    C1::flType
end

myparameters = Parameters{typeof(dt), typeof(numtraj), typeof(tspan)}(lr, epochs, numtraj,
                                                                      numtrajplot, dt,
                                                                      tinterval, tspan,
                                                                      Nintervals, ts,
                                                                      Δ, Ωmax, κ, C1)

################################################
# Define Neural Network

# state-aware
# nn = FastChain(
#   FastDense(4, 32, relu),
#   FastDense(32, 1, tanh))

# p_nn = initial_params(nn) # random initial parameters

###############################################
# initial state anywhere on the Bloch sphere
function prepare_initial(dt, n_par)
    # shape 4 x n_par
    # input number of parallel realizations and dt for type inference
    # random position on the Bloch sphere
    theta = acos.(2 * rand(typeof(dt), n_par) .- 1)  # uniform sampling for cos(theta) between -1 and 1
    phi = rand(typeof(dt), n_par) * 2 * pi  # uniform sampling for phi between 0 and 2pi
    # real and imaginary parts ceR, cdR, ceI, cdI
    u0 = [
        cos.(theta / 2),
        sin.(theta / 2) .* cos.(phi),
        false * theta,
        sin.(theta / 2) .* sin.(phi),
    ]
    return vcat(transpose.(u0)...) # build matrix
end

# target state
# ψtar = |up>

u0 = prepare_initial(myparameters.dt, myparameters.numtraj)

###############################################
# Define SDE

function qubit_drift!(du, u, p, t)
    # expansion coefficients |Ψ> = ce |e> + cd |d>
    ceR, cdR, ceI, cdI = u # real and imaginary parts

    # Δ: atomic frequency
    # Ω: Rabi frequency for field in x direction
    # κ: spontaneous emission
    Δ, Ωmax, κ = p[(end - 2):end]
    nn_weights = p[1:(end - 3)]
    Ω = (nn(u, nn_weights) .* Ωmax)[1]

    @inbounds begin
        du[1] = 1 // 2 * (ceI * Δ - ceR * κ + cdI * Ω)
        du[2] = -cdI * Δ / 2 + 1 * ceR * (cdI * ceI + cdR * ceR) * κ + ceI * Ω / 2
        du[3] = 1 // 2 * (-ceR * Δ - ceI * κ - cdR * Ω)
        du[4] = cdR * Δ / 2 + 1 * ceI * (cdI * ceI + cdR * ceR) * κ - ceR * Ω / 2
    end
    return nothing
end

function qubit_diffusion!(du, u, p, t)
    ceR, cdR, ceI, cdI = u # real and imaginary parts

    κ = p[end]

    @inbounds begin
        du[1] = zero(ceR)
        du[2] = sqrt(κ) * ceR
        du[3] = zero(ceR)
        du[4] = sqrt(κ) * ceI
    end
    return nothing
end

# normalization callback
# condition(u,t,integrator) = true
# function affect!(integrator)
#   integrator.u .= integrator.u/norm(integrator.u)
# end
# callback = DiscreteCallback(condition, affect!, save_positions=(false, false))

# CreateGrid(t,W1) = NoiseGrid(t,W1)
# Zygote.@nograd CreateGrid #avoid taking grads of this function

# set scalar random process
# W = sqrt(myparameters.dt)*randn(typeof(myparameters.dt),size(myparameters.ts)) #for 1 trajectory
# W1 = cumsum([zero(myparameters.dt); W[1:end-1]], dims=1)
# NG = CreateGrid(myparameters.ts,W1)

# get control pulses
p_nn = rand(Float32, 193)
p_all = [p_nn; myparameters.Δ; myparameters.Ωmax; myparameters.κ]

### GPU Version #####

using StaticArrays

function qubit_drift(u, p, t)
    # expansion coefficients |Ψ> = ce |e> + cd |d>
    # ceR, cdR, ceI, cdI = u # real and imaginary parts
    @inbounds begin
        ceR = u[1]
        cdR = u[2]
        ceI = u[3]
        cdI = u[4]
    end

    # @cushow u[1]
    # @cushow u[2]
    # @cushow u[3]
    # @cushow u[4]

    # Δ: atomic frequency
    # Ω: Rabi frequency for field in x direction
    # κ: spontaneous emission
    Δ = p[end - 2]
    Ωmax = p[end - 1]
    κ = p[end]
    # Δ, Ωmax, κ = p[end-2:end]
    # nn_weights = @view p[1:end-3]
    #Ω = (nn(u, nn_weights).*Ωmax)
    Ω = 3.0f0
    # # @show Ω
    du1 = 0.5f0 * (ceI * Δ - ceR * κ + cdI * Ω)
    du2 = -cdI * Δ / (2.0f0) + (1.0f0) * ceR * (cdI * ceI + cdR * ceR) * κ +
          ceI * Ω / (2.0f0)
    du3 = 0.5f0 * (-ceR * Δ - ceI * κ - cdR * Ω)
    du4 = cdR * Δ / (2.0f0) + 1 * ceI * (cdI * ceI + cdR * ceR) * κ - ceR * Ω / (2.0f0)
    # du1 = u[1]
    # du2 = u[2]
    # du3 = u[3]
    return SVector{4}(du1, du2, du3, du4)
end

function qubit_diffusion(u, p, t)
    @inbounds begin
        ceR = u[1]
        cdR = u[2]
        ceI = u[3]
        cdI = u[4]
    end
    # ceR, cdR, ceI, cdI = u # real and imaginary parts
    κ = p[end]
    du1 = zero(ceR)
    du2 = sqrt(κ)*ceR
    du3 = zero(ceR)
    du4 = sqrt(κ)*ceI
    return SVector{4, Float32}(du1, du2, du3, du4)
end

p_all_static = SArray{Tuple{length(p_all)}}(p_all)

u0_static = SArray{Tuple{length(vec(u0[:, 1]))}}(vec(u0[:, 1]))

# define SDE problem

gpu_prob = SDEProblem(qubit_drift, qubit_diffusion, u0_static, myparameters.tspan,
                      p_all_static)

monteprob = EnsembleProblem(gpu_prob)

sol = solve(monteprob, GPUEM(), EnsembleGPUKernel(), trajectories = 2, dt = myparameters.dt,
            adaptive = false)

# sol = solve(monteprob, EM(), EnsembleGPUArray(), trajectories = 2, dt = myparameters.dt)

u = gpu_prob.u0
f = gpu_prob.f
g = gpu_prob.g
p = gpu_prob.p
t = 0.0f0
test = g(u, p, t)
test2 = f(u, p, t)
sqdt = sqrt(dt)
u = u + test * dt + sqdt * test2
# 4-element SVector{4, Float32} with indices SOneTo(4):
#   0.53678507
#  -0.83391225
#  -0.09063143
#  -0.24137592

# 4-element SVector{4, Float32} with indices SOneTo(4):
#   0.50812286
#  -0.7830119
#  -0.18088597
#  -0.4440899
