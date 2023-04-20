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


Julia_data = readdlm(joinpath(base_path, "SDE", "Julia_times_unadaptive.txt"))

GPU_times = Julia_data[:, 2] .* 1e-3
Ns = Julia_data[:, 1]

CPU_data = readdlm(joinpath(base_path, "CPU", "SDE", "times_unadaptive.txt"))

CPU_times = CPU_data[:, 2] .* 1e-3

times["Fixed_CPU"] = mean(CPU_times)

times["Fixed_GPU"] = mean(GPU_times)


xticks = 10 .^ round.(range(1, 7, length = 10), digits = 2)

yticks = 10 .^ round.(range(1, -6, length = 15), digits = 2)



plt = groupedbar(
    log10.(Ns),
    [GPU_times CPU_times],
    labels = ["GPU: Float32" "CPU: Float64"],
    yaxis = :log,
    yticks = yticks,
    ylabel = "Time (s)",
    xlabel = "Trajectories (" * L"$10^n$" * ")",
    legend = :topleft,
    title = "Performance Comparison of solving SDEs \n between CPU and GPU",
    dpi = 600,
)

plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

isdir(plots_dir) || mkdir(plots_dir)


savefig(plt, joinpath(plots_dir, "CPU_SDE_$(Dates.value(Dates.now())).png"))
