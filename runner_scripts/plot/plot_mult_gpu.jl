using Plots
using DelimitedFiles
using Dates
using Statistics
using StatsPlots
using LaTeXStrings

times = Dict()

CUDA_data = readdlm("./data/quadro_RTX_5000/Julia_times_unadaptive.txt")

CUDA_times = CUDA_data[:, 2] .* 1e-3
Ns = CUDA_data[:, 1]

oneAPI_data = readdlm("./data/oneAPI/Julia_times_unadaptive.txt")

oneAPI_times = oneAPI_data[:, 2] .* 1e-3

AMDGPU_data = readdlm("./data/AMDGPU/Julia_times_unadaptive.txt")

AMDGPU_times = AMDGPU_data[:, 2] .* 1e-3

Metal_data = readdlm("./data/Metal M1 Max/Julia_times_unadaptive.txt")

Metal_times = Metal_data[:, 2] .* 1e-3

xticks = 10 .^ round.(range(1, 7, length = 10), digits = 2)

yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)

s = "Trajectories (" * L"$10^n$" * ")"

colors = collect(palette(:default))

plt = groupedbar(
    log10.(Ns),
    [CUDA_times oneAPI_times AMDGPU_times Metal_times],
    labels = ["CUDA (Nvidia Quadro RTX 5000)" "oneAPI (Intel A770)" "AMDGPU (AMD Vega 56/64)" "Metal (Apple M1 Max)"],
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

savefig(
    plt,
    joinpath(@__DIR__, "plots", "Multi_GPU_unadaptive_$(Dates.value(Dates.now())).png"),
)

CUDA_data = readdlm("./data/quadro_RTX_5000/Julia_times_adaptive.txt")

CUDA_times = CUDA_data[:, 2] .* 1e-3
Ns = CUDA_data[:, 1]

oneAPI_data = readdlm("./data/oneAPI/Julia_times_adaptive.txt")

oneAPI_times = oneAPI_data[:, 2] .* 1e-3

AMDGPU_data = readdlm("./data/AMDGPU/Julia_times_adaptive.txt")

AMDGPU_times = AMDGPU_data[:, 2] .* 1e-3

Metal_data = readdlm("./data/Metal M1 Max/Julia_times_adaptive.txt")

Metal_times = Metal_data[:, 2] .* 1e-3

plt = groupedbar(
    log10.(Ns),
    [CUDA_times oneAPI_times AMDGPU_times Metal_times],
    labels = ["CUDA (Nvidia Quadro RTX 5000)" "oneAPI (Intel A770)" "AMDGPU (AMD Vega 56/64)" "Metal (Apple M1 Max)"],
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


savefig(
    plt,
    joinpath(@__DIR__, "plots", "Multi_GPU_adaptive_$(Dates.value(Dates.now())).png"),
)
