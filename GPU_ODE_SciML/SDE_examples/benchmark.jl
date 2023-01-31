using DiffEqGPU, DiffEqBase, StaticArrays, CUDA, BenchmarkTools
trajectories = 10_000

# Defining the Problem
# dX = pudt + qudW
u₀ = SA[0.1f0]
f(u, p, t) = SA[p[1] * u[1]]
g(u, p, t) = SA[p[2] * u[1]]
tspan = (0.0f0, 1.0f0)
p = SA[1.5f0, 0.01f0]

prob = SDEProblem(f, g, u₀, tspan, p; seed = 1234)

monteprob = EnsembleProblem(prob)

## Building problem for each trajectories. Since we just want to generate different
## time-series, the problem remains same.
probs = map(1:trajectories) do i
    prob
end;

## Move the arrays to the GPU
probs = cu(probs);

## Finally use the lower API for faster solves! (Fixed time-stepping)

@benchmark CUDA.@sync ts, us = DiffEqGPU.vectorized_solve($probs, $prob, GPUEM();
                                                          save_everystep = false,
                                                          dt = Float32(1 // 2^8))

@benchmark CUDA.@sync ts, us = DiffEqGPU.vectorized_solve($probs, $prob, GPUSIEA();
                                                          save_everystep = false,
                                                          dt = Float32(1 // 2^8))
