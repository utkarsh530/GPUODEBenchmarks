using Plots
using DelimitedFiles
using Dates
using Statistics

gr(size = (720, 480))

times = Dict()

parent_dir =
    length(ARGS) != 0 ? joinpath(ARGS[1], "data") : joinpath("paper_artifacts", "data")


parent_dir = "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

if length(ARGS) != 0
    Julia_data = readdlm(joinpath(base_path, "Julia_times_unadaptive.txt"))
else
    Julia_data = readdlm(
        joinpath(
            dirname(dirname(@__DIR__)),
            parent_dir,
            "Tesla_V100",
            "Julia",
            "Julia_times_unadaptive.txt",
        ),
    )
end

Julia_data = readdlm(joinpath(base_path, "Julia", "stiff", "Julia_times_adaptive.txt"))

GPU_times = Julia_data[:, 2] .* 1e-3
Ns = Julia_data[:, 1]

Julia_EGArray_data = readdlm(
    joinpath(base_path, "EnsembleGPUArray", "stiff", "Julia_EnGPUArray_times_adaptive.txt"),
)

GPU_EGArray_times = Julia_EGArray_data[:, 2] .* 1e-3

CPU_data = readdlm(joinpath(base_path, "CPU", "stiff", "Julia_times_adaptive.txt"))

CPU_times = CPU_data[:, 2] .* 1e-3

times["Fixed_CPU"] = mean(CPU_times ./ GPU_times)

times["Fixed_GPU"] = mean(GPU_times ./ GPU_times)

times["Fixed_GPU_vmap"] = mean(GPU_EGArray_times ./ GPU_times)

xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)

yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)

plt = plot(
    Ns,
    GPU_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "EnsembleGPUKernel",
    ylabel = "Time (s)",
    xlabel = "Trajectories",
    title = "Bechmarking the ROBER Problem",
    legend = :topleft,
    xticks = xticks,
    yticks = yticks,
    marker = :circle,
    dpi = 600,
    color = :Green,
)


plt = plot!(
    Ns,
    CPU_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "CPU",
    marker = :circle,
    color = :Orange,
)

plt = plot!(
    Ns,
    GPU_EGArray_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "EnsembleGPUArray",
    marker = :circle,
    color = :Red,
)


plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

isdir(plots_dir) || mkdir(plots_dir)



savefig(plt, joinpath(plots_dir, "CPU_Rober_adaptive_$(Dates.value(Dates.now())).png"))
