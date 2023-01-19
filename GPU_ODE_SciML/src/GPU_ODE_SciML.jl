module GPU_ODE_SciML
using ModelingToolkit, StaticArrays, SciMLBase

greet() = print("Hello World!")

function make_gpu_compatible(prob::T, ::Val{T1}) where {T <: ODEProblem, T1}
    sys = modelingtoolkitize(prob)
    prob = ODEProblem{false}(sys)
    remake(prob; u0 = SArray{Tuple{length(prob.u0)}, T1}(prob.u0),
           tspan = T1.(prob.tspan),
           p = prob.p isa SciMLBase.NullParameters ? prob.p :
               SArray{Tuple{length(prob.p)}, T1}(prob.p))
end

export make_gpu_compatible

end # module GPU_ODE_SciML
