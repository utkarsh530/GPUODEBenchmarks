using DiffEqGPU, BenchmarkTools, StaticArrays, StochasticDiffEq

@show ARGS
#settings

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

# Defining the Problem
# dX = pudt + qudW
u₀ = SA[0.1, 0.1, 0.1]
f(u, p, t) = p[1] * u
g(u, p, t) = p[2] * u
tspan = (0.00, 1.0)
p = SA[1.5, 0.01]

prob = SDEProblem(f, g, u₀, tspan, p; seed = 1234)

ensembleProb = EnsembleProblem(prob)

@info "Solving the problem"

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

data = @benchmark solve($ensembleProb, EM(), EnsembleThreads(), dt = Float64(1 // 2^8),
                        adaptive = false, save_everystep = false,
                        trajectories = numberOfParameters)

if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "CPU/SDE/times_unadaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
