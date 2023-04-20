using Plots
using DelimitedFiles
using Dates
using Statistics
using StatsPlots
using LaTeXStrings

times = Dict()

parent_dir =
    length(ARGS) != 0 ? joinpath(ARGS[1], "data") :
    joinpath("paper_artifacts", "data", "Julia")

base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir, "devices")

CUDA_data = readdlm(joinpath(base_path, "CUDA", "Julia_times_unadaptive.txt"))

CUDA_times = CUDA_data[:, 2] .* 1e-3
Ns = CUDA_data[:, 1]

oneAPI_data = readdlm(joinpath(base_path, "oneAPI", "Julia_times_unadaptive.txt"))

oneAPI_times = oneAPI_data[:, 2] .* 1e-3

AMDGPU_data = readdlm(joinpath(base_path, "AMDGPU", "Julia_times_unadaptive.txt"))

AMDGPU_times = AMDGPU_data[:, 2] .* 1e-3

Metal_data = readdlm(joinpath(base_path, "Metal", "Julia_times_unadaptive.txt"))

Metal_times = Metal_data[:, 2] .* 1e-3

xticks = 10 .^ round.(range(1, 7, length = 10), digits = 2)

yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)

s = "Trajectories (" * L"$10^n$" * ")"

colors = collect(palette(:default))

plt = groupedbar(
    log10.(Ns),
    [CUDA_times oneAPI_times AMDGPU_times Metal_times],
    labels = ["CUDA" "oneAPI" "AMDGPU" "Metal"],
    yaxis = :log,
    yticks = yticks,
    ylabel = "Time (s)",
    xlabel = s,
    legend = :topleft,
    title = "Performance Comparison with different GPU backends",
    titlefontsize = 12,
    palette = [colors[3], colors[1], colors[2], colors[4]],
    dpi = 300,
)

plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

isdir(plots_dir) || mkdir(plots_dir)

savefig(plt, joinpath(plots_dir, "Multi_GPU_unadaptive_$(Dates.value(Dates.now())).png"))
