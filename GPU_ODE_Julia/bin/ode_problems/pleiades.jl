function f!(du::AbstractArray{T}, u::AbstractArray{T}, p, t::T) where {T}
    @inbounds begin
        x = view(u, 1:7)   # x
        y = view(u, 8:14)  # y
        v = view(u, 15:21) # x′
        w = view(u, 22:28) # y′
        du[1:7] .= v
        du[8:14] .= w
        for i in 15:28
            du[i] = zero(u[1])
        end
        for i in 1:7, j in 1:7
            if i != j
                r = ((x[i] - x[j])^(2.0f0) + (y[i] - y[j])^(2.0f0))^(3.0f0 / 2.0f0)
                du[14 + i] += j * (x[j] - x[i]) / r
                du[21 + i] += j * (y[j] - y[i]) / r
            end
        end
    end
    du = T.(du)
end

u0 = T[3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0, 3.0, -3.0, 2.0, 0, 0, -4.0, 4.0, 0, 0, 0, 0, 0,
       1.75, -1.5, 0, 0, 0, -1.25, 1, 0, 0]
tspan = (0.0, 3.0)
oprob = ODEProblem(f!, u0, T.(tspan))

prob = make_gpu_compatible(oprob, Val(T))

@assert prob.f(prob.u0, prob.p, T(1.0)) isa StaticArray{<:Tuple, T}

ensembleProb = EnsembleProblem(prob)
dt = T(0.001)

# sol = solve(ensembleProb, GPUTsit5(), EnsembleGPUKernel(), trajectories = 2, dt = 1.0f0)

# ### Lower level API ####

## Building problems here only
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

## Make them compatible with CUDA
probs = cu(probs)

# @info "Solving the problem"
# sol = @time @CUDA.sync DiffEqGPU.vectorized_asolve(probs, ensembleProb.prob, GPUTsit5();
# save_everystep = false, dt = 0.001f0)
