"""
Scaling GPU ODE solvers to mulitple GPU cluster nodes with MPI.

Created by: Utkarsh
Last Modified: 20 April 2023
"""

using MPI
using CUDA
using DiffEqGPU, StaticArrays, CUDA, DiffEqBase
using BenchmarkTools

function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i = 1:n]
end


MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)

root = 0

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)

function perform_ode_solve(prob, parameter)
    trajectories = length(parameter)
    probs = map(1:trajectories) do i
        remake(prob, p = @SVector [10.0f0, parameter[i], 8 / 3.0f0])
    end

    ## Move the arrays to the GPU
    probs = cu(probs)

    ts, us = DiffEqGPU.vectorized_asolve(
        probs,
        prob,
        GPUTsit5();
        saveat = [prob.tspan[2]],
        dt = 0.1f0,
    )
end

if rank == root
    M, N = 1, 2^30

    test = collect(LinRange(0.0f0, 21.0f0, N))
    output = CuArray{typeof(u0)}(undef, (1, N))

    N_counts = split_count(N, comm_size - 1)

    sizes = pushfirst!(N_counts, 0)
    size_ubuf = UBuffer(sizes, 1)

    counts = sizes

    test_vbuf = VBuffer(test, counts) # VBuffer for scatter
    output_vbuf = VBuffer(output, counts) # VBuffer for gather
else
    # these variables can be set to `nothing` on non-root processes
    size_ubuf = UBuffer(nothing)
    output_vbuf = test_vbuf = VBuffer(nothing)
end

MPI.Barrier(comm)

local_size = MPI.Scatter(size_ubuf, NTuple{1,Int}, root, comm)
local_test = MPI.Scatterv!(test_vbuf, zeros(Float32, local_size), root, comm)

if rank != root
    ts, us = perform_ode_solve(prob, local_test)
else
    us = CuArray{typeof(u0)}(undef, (1, 0))
end

MPI.Barrier(comm)

@show MPI.Get_processor_name(), size(us)

MPI.Gatherv!(us, output_vbuf, root, comm)

MPI.Barrier(comm)

if rank == root
    println()
    println("Final matrix")
    println("================")
    @show size(output)
end
