using Plots
using DelimitedFiles
using Dates
using Statistics

using Plots.PlotMeasures


parent_dir =
    length(ARGS) != 0 ? joinpath(ARGS[1], "data") :
    joinpath("paper_artifacts", "data", "RTX_5000")

base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

times_v100 = Dict()

Julia_data = readdlm(joinpath(base_path, "Julia", "Julia_times_unadaptive.txt"))

Julia_times = Julia_data[:, 2] .* 1e-3
Ns = Julia_data[:, 1]

MPGOS_data = readdlm(joinpath(base_path, "MPGOS", "MPGOS_times_unadaptive.txt"))

MPGOS_times = MPGOS_data[:, 2] .* 1e-3

JAX_data = readdlm(joinpath(base_path, "JAX", "Jax_times_unadaptive.txt"))

JAX_times = JAX_data[:, 2] .* 1e-3

Torch_data = readdlm(joinpath(base_path, "PyTorch", "Torch_times_unadaptive.txt"))

Torch_times = Torch_data[:, 2] .* 1e-3

times_v100["Fixed_Julia"] =
    (minimum(Julia_times ./ Julia_times), maximum(Julia_times ./ Julia_times))

times_v100["Fixed_JAX"] =
    (minimum(JAX_times ./ Julia_times), maximum(JAX_times ./ Julia_times))

times_v100["Fixed_MPGOS"] =
    (minimum(MPGOS_times ./ Julia_times), maximum(MPGOS_times ./ Julia_times))

times_v100["Fixed_Torch"] =
    (minimum(Torch_times ./ Julia_times), maximum(Torch_times ./ Julia_times))

xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)

yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)
gr(size = (810, 540))
plt = plot(
    Ns,
    Julia_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "Julia",
    ylabel = "Time (s)",
    xlabel = "Trajectories",
    title = "Lorenz Problem: 1000 fixed time-steps",
    legend = :topleft,
    xticks = xticks,
    yticks = yticks,
    color = :Green,
    marker = :circle,
    dpi = 600,
    # left_margin = mm, bottom_margin = 4mm,top_margin = 6mm,right_margin = 6mm
)

plt = plot!(
    Ns,
    MPGOS_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "MPGOS",
    color = :Orange,
    marker = :circle,
)

plt = plot!(
    Ns,
    JAX_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "JAX",
    color = :Red,
    marker = :circle,
)

plt = plot!(
    Ns,
    Torch_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "PyTorch",
    color = :DarkRed,
    marker = :circle,
)

plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

isdir(plots_dir) || mkdir(plots_dir)


savefig(plt, joinpath(plots_dir, "Lorenz_unadaptive_$(Dates.value(Dates.now())).png"))
