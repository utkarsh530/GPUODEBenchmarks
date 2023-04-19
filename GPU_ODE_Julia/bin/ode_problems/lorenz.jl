function lorenz(u::AbstractArray{T}, p, t) where {T}
    du1 = T(10.0) * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - T(8 // 3) * u[3]
    return @SVector T[du1, du2, du3]
end

u0 = @SVector T[1.0f0; 0.0f0; 0.0f0]
tspan = (T(0.0), T(1.0))
p = @SArray T[28.0]
prob = ODEProblem(lorenz, u0, tspan, p)

parameterList = range(T(0.0), stop = T(21.0), length = numberOfParameters)

lorenzProblem = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(lorenzProblem, prob_func = prob_func)

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
dt = T(0.001)
