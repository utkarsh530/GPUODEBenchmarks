using TestEnv
using Pkg

Pkg.add("DiffEqGPU")
TestEnv.activate("DiffEqGPU")
backend = ARGS[1]
ENV["GROUP"] = backend
Pkg.add(backend)

ENV["JULIA_LOAD_PATH"]=dirname(Base.active_project())

using DiffEqGPU
include(joinpath(dirname(pathof(DiffEqGPU)), "..", "test", "runtests.jl"))
