using Plots
using DelimitedFiles
using Dates

Julia_data = readdlm("./data/quadro_RTX_5000/Julia_times_unadaptive.txt")

Julia_times = Julia_data[:, 2] .* 1e-3
Ns = Julia_data[:, 1]

MPGOS_data = readdlm("./data/quadro_RTX_5000/MPGOS_times_unadaptive.txt")

MPGOS_times = MPGOS_data[:, 2] .* 1e-3

JAX_data = readdlm("./data/quadro_RTX_5000/Jax_times_unadaptive.txt")

JAX_times = JAX_data[:, 2] .* 1e-3


xticks = 10 .^ round.(range(1, 7, length = 10), digits = 2)

yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)

plt = plot(
    Ns,
    Julia_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "GPUTsit5",
    ylabel = "Time (s)",
    xlabel = "Trajectories",
    title = "Lorenz Problem: 1000 fixed time-steps",
    legend = :topleft,
    xticks = xticks,
    yticks = yticks,
)

plt = plot!(Ns, MPGOS_times, xaxis = :log, yaxis = :log, linewidth = 2, label = "MPGOS")

plt = plot!(Ns, JAX_times, xaxis = :log, yaxis = :log, linewidth = 2, label = "JAX")

savefig(plt, joinpath(@__DIR__, "plots", "Lorenz_unadaptive_$(Dates.now()).png"))


Julia_data = readdlm("./data/quadro_RTX_5000/Julia_times_adaptive.txt")

Julia_times = Julia_data[:, 2] .* 1e-3
Ns = Julia_data[:, 1]

MPGOS_data = readdlm("./data/quadro_RTX_5000/MPGOS_times_adaptive.txt")

MPGOS_times = MPGOS_data[:, 2] .* 1e-3

JAX_data = readdlm("./data/quadro_RTX_5000/Jax_times_adaptive.txt")

JAX_times = JAX_data[:, 2] .* 1e-3

xticks = 10 .^ round.(range(1, 7, length = 10), digits = 2)

yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)

plt = plot(
    Ns,
    Julia_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "GPUTsit5",
    ylabel = "Time (s)",
    xlabel = "Trajectories",
    title = "Lorenz Problem: Adaptive time-stepping",
    legend = :topleft,
    xticks = xticks,
    yticks = yticks,
)

plt = plot!(Ns, MPGOS_times, xaxis = :log, yaxis = :log, linewidth = 2, label = "MPGOS")

plt = plot!(Ns, JAX_times, xaxis = :log, yaxis = :log, linewidth = 2, label = "JAX")

savefig(plt, joinpath(@__DIR__, "plots", "Lorenz_adaptive_$(Dates.now()).png"))
