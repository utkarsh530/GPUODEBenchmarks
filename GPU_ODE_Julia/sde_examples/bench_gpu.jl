using DiffEqGPU, DiffEqBase, StaticArrays, CUDA, BenchmarkTools

@show ARGS
#settings

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

# Defining the Problem
# dX = pudt + qudW
u₀ = SA[0.1f0, 0.1f0, 0.1f0]
f(u, p, t) = p[1] * u
g(u, p, t) = p[2] * u
tspan = (0.0f0, 1.0f0)
p = SA[1.5f0, 0.01f0]

prob = SDEProblem(f, g, u₀, tspan, p; seed = 1234)

ensembleProb = EnsembleProblem(prob)

## Building problem for each trajectories. Since we just want to generate different
## time-series, the problem remains same.
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

## Move the arrays to the GPU
probs = cu(probs);

## Finally use the lower API for faster solves! (Fixed time-stepping)

data = @benchmark CUDA.@sync DiffEqGPU.vectorized_solve($probs, $prob, GPUEM();
                                                        save_everystep = false,
                                                        dt = Float32(1 // 2^8))

if !isinteractive()
    open(joinpath(dirname(dirname(@__DIR__)), "data", "SDE", "Julia_times_unadaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
