using Random
Random.seed!(123)

# 1D Linear ODE
function f(u::AbstractArray{T}, p, t::T) where {T}
    return T(1.01) * u
end
function f_analytic(u₀, p, t)
    u₀ * exp(1.01 * t)
end

tspan = (0.0, 10.0)
tspan = T.(tspan)
u0 = @SVector rand(T, 100)
prob = ODEProblem(ODEFunction(f, analytic = f_analytic), u0, tspan)

ensembleProb = EnsembleProblem(prob)

### Lower level API ####

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
dt = T(0.1)
