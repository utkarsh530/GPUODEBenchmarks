using Plots
using DelimitedFiles
using Dates
using Statistics

gr(size = (720, 480))

times = Dict()

parent_dir =
    length(ARGS) != 0 ? joinpath(ARGS[1], "data") : joinpath("paper_artifacts", "data")

base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir, "Julia")

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

GPU_times = Julia_data[:, 2][1:9] .* 1e-3
Ns = Julia_data[:, 1][1:9]

Julia_EGArray_data = readdlm(
    joinpath(base_path, "EnsembleGPUArray", "Julia_EnGPUArray_times_unadaptive.txt"),
)

GPU_EGArray_times = Julia_EGArray_data[:, 2][1:9] .* 1e-3

CPU_data = readdlm(joinpath(base_path, "CPU", "times_unadaptive.txt"))

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
    label = "EnsembleGPUKernel: Fixed dt",
    ylabel = "Time (s)",
    xlabel = "Trajectories",
    title = "Bechmarking the Lorenz Problem",
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
    label = "CPU: Fixed dt",
    marker = :circle,
    color = :Orange,
)

plt = plot!(
    Ns,
    GPU_EGArray_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "EnsembleGPUArray: Fixed dt",
    marker = :circle,
    color = :Red,
)


plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

isdir(plots_dir) || mkdir(plots_dir)


if length(ARGS) != 0
    Julia_data = readdlm(joinpath(base_path, "Julia_times_adaptive.txt"))
else
    Julia_data = readdlm(
        joinpath(
            dirname(dirname(@__DIR__)),
            parent_dir,
            "Tesla_V100",
            "Julia",
            "Julia_times_adaptive.txt",
        ),
    )
end

GPU_times = Julia_data[:, 2][1:9] .* 1e-3
Ns = Julia_data[:, 1][1:9]

Julia_EGArray_data =
    readdlm(joinpath(base_path, "EnsembleGPUArray", "Julia_EnGPUArray_times_adaptive.txt"))

GPU_EGArray_times = Julia_EGArray_data[:, 2][1:9] .* 1e-3

CPU_data = readdlm(joinpath(base_path, "CPU", "times_unadaptive.txt"))

CPU_times = CPU_data[:, 2] .* 1e-3

times["Adaptive_CPU"] = mean(CPU_times ./ GPU_times)

times["Adaptive_GPU"] = mean(GPU_times ./ GPU_times)

times["Adaptive_GPU_vmap"] = mean(GPU_EGArray_times ./ GPU_times)


plt = plot!(
    Ns,
    GPU_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    marker = :ltriangle,
    dpi = 600,
    color = :Green,
    label = "EnsembleGPUKernel: Adaptive dt",
)

plt = plot!(
    Ns,
    CPU_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "CPU: Adaptive dt",
    marker = :ltriangle,
    color = :Orange,
)

plt = plot!(
    Ns,
    GPU_EGArray_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "EnsembleGPUArray: Adaptive dt",
    marker = :ltriangle,
    color = :Red,
)

savefig(plt, joinpath(plots_dir, "CPU_Lorenz_adaptive_$(Dates.value(Dates.now())).png"))
