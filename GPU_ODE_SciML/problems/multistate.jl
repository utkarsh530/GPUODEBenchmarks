using ReactionNetworkImporters, Catalyst

prnbng = loadrxnetwork(BNGNetwork(), joinpath(dirname(@__DIR__), "Models/multistate.net"))

rn = prnbng.rn
obs = [eq.lhs for eq in observed(rn)]

osys = convert(ODESystem, rn)

tf = 20.0
tspan = (0.0, tf)
oprob = ODEProblem{false}(osys, T[], tspan, T[])

prob = make_gpu_compatible(oprob, Val(T))

@assert prob.f(prob.u0, prob.p, T(1.0f0)) isa StaticArray{<:Tuple, T}

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
dt = T(0.001)
