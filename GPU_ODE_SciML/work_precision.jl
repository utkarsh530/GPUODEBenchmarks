using GPU_ODE_SciML
using DiffEqGPU, BenchmarkTools, StaticArrays, SimpleDiffEq, OrdinaryDiffEq
using CUDA
using BenchmarkTools
using DiffEqDevTools
using Plots, Dates

@show ARGS
#settings

numberOfParameters = isinteractive() ? 1024 : parse(Int64, ARGS[1])
filename = isinteractive() ? "multistate.jl" : ARGS[2]
gpuID = 0

@show device!(CuDevice(gpuID))

problems_path = joinpath(@__DIR__, "problems")

## Set type for problem. Performance on GPUs greatly affects with Floats.
T = Float32

include(joinpath(problems_path, filename));

## Comparsion of primitive solve vs wrapper solve

## 61.375 ms
# @benchmark CUDA.@sync solve($ensembleProb, GPUTsit5(), EnsembleGPUKernel(0.0),
#                                        trajectories = numberOfParameters, adaptive = true,
#                                        save_everystep = false,
#                                        dt = dt)

## 61.195 ms
# @benchmark CUDA.@sync solve($prob, $(GPUODE{GPUTsit5}(numberOfParameters)), adaptive = true, save_everystep = false, dt = dt)

abstols = 1.0f0 ./ 10.0f0 .^ (4:7)
reltols = 1.0f0 ./ 10.0f0 .^ (1:4)

sol = solve(prob, Vern9(), abstol = 1.0f-7, reltol = 1.0f-7, maxiters = 1000000)

test_sol = TestSolution(sol);

setups = [Dict(:alg => GPUODE{GPUTsit5}(numberOfParameters))
          Dict(:alg => GPUODE{GPUVern7}(numberOfParameters))
          Dict(:alg => GPUODE{GPUVern9}(numberOfParameters))]

@info @show numberOfParameters

@info "Generating WP for Float32"
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
                      save_everystep = false, dt = dt,
                      names = ["GPUTsit5", "GPUVern7", "GPUVern9"])
plt = plot(wp);
savefig(plt,
        joinpath(@__DIR__, "wp_diagrams",
                 "wp_$(split(filename, ".")[1])_gpu_float32_$(Dates.value(Dates.now())).png"));

## Float64
T = Float64

include(joinpath(problems_path, filename));

sol = solve(prob, Vern9(), abstol = 1 / 10^12, reltol = 1 / 10^12, maxiters = 1000000)

test_sol = TestSolution(sol);
abstols = 1.0 ./ 10.0 .^ (6:10)
reltols = 1.0 ./ 10.0 .^ (3:7)

@info @show numberOfParameters
@info "Generating WP for Float64"
wp = WorkPrecisionSet(prob, abstols, reltols, setups; appxsol = test_sol,
                      save_everystep = false, dt = dt,
                      names = ["GPUTsit5", "GPUVern7", "GPUVern9"])
plt = plot(wp);
savefig(plt,
        joinpath(@__DIR__, "wp_diagrams",
                 "wp_$(split(filename, ".")[1])_gpu_float64_$(Dates.value(Dates.now())).png"));
