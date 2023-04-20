using Plots
using DelimitedFiles
using Dates
using Statistics
using LaTeXStrings
using StatsPlots

times = Dict()

parent_dir =
    length(ARGS) != 0 ? joinpath(ARGS[1], "data") :
    joinpath("paper_artifacts", "data", "Julia")

base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

Julia_data = readdlm(joinpath(base_path, "SDE", "CRN", "Julia_times_unadaptive.txt"))

GPU_times = Julia_data[:, 2] .* 1e-3
Ns = Julia_data[:, 1]

CPU_data = readdlm(joinpath(base_path, "CPU", "SDE", "CRN", "Julia_times_unadaptive.txt"))

CPU_times = CPU_data[:, 2] .* 1e-3

times["Fixed_CPU"] = mean(CPU_times)

times["Fixed_GPU"] = mean(GPU_times)


xticks = 10 .^ round.(range(1, 7, length = 10), digits = 2)

yticks = 10 .^ round.(range(2, -3, length = 11), digits = 2)

# plt = plot(
#     Ns,
#     GPU_times,
#     xaxis = :log,
#     yaxis = :log,
#     linewidth = 2,
#     label = "GPU: Float32",
#     ylabel = "Time (s)",
#     xlabel = "Trajectories",
#     title = "Lorenz Problem: 1000 fixed time-steps",
#     legend = :topleft,
#     xticks = xticks,
#     yticks = yticks,
#     marker = :circle,
# )



plt = groupedbar(
    log10.(Ns),
    [GPU_times CPU_times],
    labels = ["GPU" "CPU"],
    yaxis = :log,
    yticks = yticks,
    ylabel = "Time (s)",
    xlabel = "Trajectories (" * L"$10^n$" * ")",
    legend = :topleft,
    title = "Performance Comparison of parallel-parameter \n sweeps in SDEs between CPU and GPU",
    dpi = 600,
)


plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

isdir(plots_dir) || mkdir(plots_dir)

savefig(plt, joinpath(plots_dir, "CPU_SDE_CRN_$(Dates.value(Dates.now())).png"))
