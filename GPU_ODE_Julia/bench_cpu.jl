using BenchmarkTools, StaticArrays, OrdinaryDiffEq

@show ARGS
#settings

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

function lorenz(u, p, t)
    du1 = 10.0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - 2.666 * u[3]
    return @SVector [du1, du2, du3]
end

u0 = @SVector [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
p = @SArray [21.0]
prob = ODEProblem(lorenz, u0, tspan, p)

parameterList = range(0.0, stop = 21.0, length = numberOfParameters)

lorenzProblem = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(lorenzProblem, prob_func = prob_func)

@info "Solving the problem"
data = @benchmark solve($ensembleProb, Tsit5(), EnsembleThreads(), dt = 0.001,
                        adaptive = false, save_everystep = false,
                        trajectories = numberOfParameters)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "CPU", "times_unadaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

data = @benchmark solve($ensembleProb, Tsit5(), EnsembleThreads(), dt = 0.001,
                        adaptive = true, save_everystep = false,
                        trajectories = numberOfParameters)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "CPU", "times_adaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
