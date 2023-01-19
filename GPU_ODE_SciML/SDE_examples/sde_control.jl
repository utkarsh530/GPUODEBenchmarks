# load packages
using DiffEqFlux
using SciMLSensitivity
using Optimization
using StochasticDiffEq, DiffEqCallbacks, DiffEqNoiseProcess
using Statistics, LinearAlgebra, Random
using Plots

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
nn = FastChain(FastDense(4, 32, relu),
               FastDense(32, 1, tanh))

p_nn = initial_params(nn) # random initial parameters

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

    du .= false

    @inbounds begin
        #du[1] = zero(ceR)
        du[2] += sqrt(κ) * ceR
        #du[3] = zero(ceR)
        du[4] += sqrt(κ) * ceI
    end
    return nothing
end

# normalization callback
condition(u, t, integrator) = true
function affect!(integrator)
    integrator.u .= integrator.u / norm(integrator.u)
end
callback = DiscreteCallback(condition, affect!, save_positions = (false, false))

CreateGrid(t, W1) = NoiseGrid(t, W1)
Zygote.@nograd CreateGrid #avoid taking grads of this function

# set scalar random process
W = sqrt(myparameters.dt) * randn(typeof(myparameters.dt), size(myparameters.ts)) #for 1 trajectory
W1 = cumsum([zero(myparameters.dt); W[1:(end - 1)]], dims = 1)
NG = CreateGrid(myparameters.ts, W1)

# get control pulses
p_all = [p_nn; myparameters.Δ; myparameters.Ωmax; myparameters.κ]
# define SDE problem

# prob = SDEProblem{true}(qubit_drift!, qubit_diffusion!, vec(u0[:, 1]), myparameters.tspan,
#                         p_all,
#                         callback = callback, noise = NG)

prob = SDEProblem{true}(qubit_drift!, qubit_diffusion!, vec(u0[:, 1]), myparameters.tspan,
                        p_all)

#########################################
# compute loss
function g(u, p, t)
    ceR = @view u[1, :, :]
    cdR = @view u[2, :, :]
    ceI = @view u[3, :, :]
    cdI = @view u[4, :, :]
    p[1] * mean((cdR .^ 2 + cdI .^ 2) ./ (ceR .^ 2 + cdR .^ 2 + ceI .^ 2 + cdI .^ 2))
end

p = p_nn
alg = EM()
pars = [p; myparameters.Δ; myparameters.Ωmax; myparameters.κ]
u0 = prepare_initial(myparameters.dt, myparameters.numtraj)

function prob_func(prob, i, repeat)
    # prepare initial state and applied control pulse
    u0tmp = deepcopy(vec(u0[:, i]))
    W = sqrt(myparameters.dt) * randn(typeof(myparameters.dt), size(myparameters.ts)) #for 1 trajectory
    W1 = cumsum([zero(myparameters.dt); W[1:(end - 1)]], dims = 1)
    NG = CreateGrid(myparameters.ts, W1)

    remake(prob,
            p = pars,
            u0 = u0tmp,
            callback = callback,
            noise = NG)
end

ensembleprob = EnsembleProblem(prob,
                                prob_func = prob_func,
                                safetycopy = true)

_sol = solve(ensembleprob, alg, EnsembleThreads(),
                sensealg = sensealg,
                saveat = myparameters.tinterval,
                dt = myparameters.dt,
                adaptive = false,
                trajectories = myparameters.numtraj, batch_size = myparameters.numtraj)


# function loss(p; alg = EM(), sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP()))
#     pars = [p; myparameters.Δ; myparameters.Ωmax; myparameters.κ]
#     u0 = prepare_initial(myparameters.dt, myparameters.numtraj)

#     function prob_func(prob, i, repeat)
#         # prepare initial state and applied control pulse
#         u0tmp = deepcopy(vec(u0[:, i]))
#         W = sqrt(myparameters.dt) * randn(typeof(myparameters.dt), size(myparameters.ts)) #for 1 trajectory
#         W1 = cumsum([zero(myparameters.dt); W[1:(end - 1)]], dims = 1)
#         NG = CreateGrid(myparameters.ts, W1)

#         remake(prob,
#                p = pars,
#                u0 = u0tmp,
#                callback = callback,
#                noise = NG)
#     end

#     ensembleprob = EnsembleProblem(prob,
#                                    prob_func = prob_func,
#                                    safetycopy = true)

#     _sol = solve(ensembleprob, alg, EnsembleThreads(),
#                  sensealg = sensealg,
#                  saveat = myparameters.tinterval,
#                  dt = myparameters.dt,
#                  adaptive = false,
#                  trajectories = myparameters.numtraj, batch_size = myparameters.numtraj)
#     A = convert(Array, _sol)

#     l = g(A, [myparameters.C1], nothing)
#     # returns loss value
#     return l
# end

# #########################################
# # visualization -- run for new batch
# function visualize(p; alg = EM())
#     u0 = prepare_initial(myparameters.dt, myparameters.numtrajplot)
#     pars = [p; myparameters.Δ; myparameters.Ωmax; myparameters.κ]

#     function prob_func(prob, i, repeat)
#         # prepare initial state and applied control pulse
#         u0tmp = deepcopy(vec(u0[:, i]))
#         W = sqrt(myparameters.dt) * randn(typeof(myparameters.dt), size(myparameters.ts)) #for 1 trajectory
#         W1 = cumsum([zero(myparameters.dt); W[1:(end - 1)]], dims = 1)
#         NG = CreateGrid(myparameters.ts, W1)

#         remake(prob,
#                p = pars,
#                u0 = u0tmp,
#                callback = callback,
#                noise = NG)
#     end

#     ensembleprob = EnsembleProblem(prob,
#                                    prob_func = prob_func,
#                                    safetycopy = true)

#     u = solve(ensembleprob, alg, EnsembleThreads(),
#               saveat = myparameters.tinterval,
#               dt = myparameters.dt,
#               adaptive = false, #abstol=1e-6, reltol=1e-6,
#               trajectories = myparameters.numtrajplot,
#               batch_size = myparameters.numtrajplot)

#     ceR = @view u[1, :, :]
#     cdR = @view u[2, :, :]
#     ceI = @view u[3, :, :]
#     cdI = @view u[4, :, :]
#     infidelity = @. (cdR^2 + cdI^2) / (ceR^2 + cdR^2 + ceI^2 + cdI^2)
#     meaninfidelity = mean(infidelity)
#     loss = myparameters.C1 * meaninfidelity

#     @info "Loss: " loss

#     fidelity = @. (ceR^2 + ceI^2) / (ceR^2 + cdR^2 + ceI^2 + cdI^2)

#     mf = mean(fidelity, dims = 2)[:]
#     sf = std(fidelity, dims = 2)[:]

#     pl1 = plot(0:(myparameters.Nintervals), mf,
#                ribbon = sf,
#                ylim = (0, 1), xlim = (0, myparameters.Nintervals),
#                c = 1, lw = 1.5, xlabel = "steps i", ylabel = "Fidelity", legend = false)

#     pl = plot(pl1, legend = false, size = (400, 360))
#     return pl, loss
# end

# # burn-in loss
# l = loss(p_nn)
# # callback to visualize training
# visualization_callback = function (p, l; doplot = false)
#     println(l)

#     if doplot
#         pl, _ = visualize(p)
#         display(pl)
#     end

#     return false
# end

# # Display the ODE with the initial parameter values.
# visualization_callback(p_nn, l; doplot = true)

# ###################################
# # training loop
# @info "Start Training.."

# # optimize the parameters for a few epochs with ADAM on time span
# # Setup and run the optimization
# adtype = Optimization.AutoZygote()
# optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)

# optprob = Optimization.OptimizationProblem(optf, p_nn)
# res = Optimization.solve(optprob, ADAM(myparameters.lr), callback = visualization_callback,
#                          maxiters = 100)

# # plot optimized control
# visualization_callback(res.u, loss(res.u); doplot = true)
