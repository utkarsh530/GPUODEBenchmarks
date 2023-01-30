module GPU_ODE_SciML
using ModelingToolkit, StaticArrays, SciMLBase
using DiffEqGPU

greet() = print("Hello World!")

function make_gpu_compatible(prob::T, ::Val{T1}) where {T <: ODEProblem, T1}
    sys = modelingtoolkitize(prob)
    prob = ODEProblem{false}(sys)
    remake(prob; u0 = SArray{Tuple{length(prob.u0)}, T1}(prob.u0),
           tspan = T1.(prob.tspan),
           p = prob.p isa SciMLBase.NullParameters ? prob.p :
               SArray{Tuple{length(prob.p)}, T1}(prob.p))
end

struct GPUODE{T <: DiffEqGPU.GPUODEAlgorithm} <: SciMLBase.AbstractODEAlgorithm
    trajectories::Int
end

## Wrapping for compat with WorkPrecisionSet
function SciMLBase.__solve(prob::SciMLBase.AbstractODEProblem, alg::GPUODE{T}, args...;
                           kwargs...) where {T}
    eprob = EnsembleProblem(prob)
    sol = solve(eprob, T(), EnsembleGPUKernel(0.0), trajectories = alg.trajectories;
                kwargs...)
    return sol[1]
end

export make_gpu_compatible, GPUODE

end # module GPU_ODE_SciML
