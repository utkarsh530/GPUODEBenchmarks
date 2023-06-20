using Plots
using DelimitedFiles
using Dates
using Statistics

using Plots.PlotMeasures


parent_dir = "data"

base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

times_v100 = Dict()

Julia_data_rb = readdlm(joinpath(base_path, "Julia", "stiff", "Julia_times_adaptive_rb.txt"))

Julia_times_rb = Julia_data_rb[:, 2] .* 1e-3
Ns = Julia_data_rb[:, 1]

Julia_data_rs5p = readdlm(joinpath(base_path, "Julia", "stiff", "Julia_times_adaptive_rs5p.txt"))

Julia_times_rs5p = Julia_data_rs5p[:, 2] .* 1e-3


Julia_data_rs4 = readdlm(joinpath(base_path, "Julia", "stiff", "Julia_times_adaptive_rs4.txt"))

Julia_times_rs4 = Julia_data_rs4[:, 2] .* 1e-3


Julia_EGArray_kv5_data = readdlm(joinpath(base_path, "EnsembleGPUArray", "stiff", "Julia_EnGPUArray_kv5_times_adaptive.txt"))

Julia_EGArray_kv5_times = Julia_EGArray_kv5_data[:, 2] .* 1e-3

Julia_EGArray_rs5p_data = readdlm(joinpath(base_path, "EnsembleGPUArray", "stiff", "Julia_EnGPUArray_rs5p_times_adaptive.txt"))

Julia_EGArray_rs5p_times = Julia_EGArray_rs5p_data[:, 2] .* 1e-3

JAX_data = readdlm(joinpath(base_path, "JAX", "stiff", "Jax_times_adaptive.txt"))

JAX_times = JAX_data[:, 2] .* 1e-3

xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)

yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)


gr(size = (600, 400))
plt = plot(
    Ns,
    Julia_times_rb,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "Julia: Rosenbrock23",
    ylabel = "Time (s)",
    xlabel = "Trajectories",
    title = "ROBER Problem, low tolerances",
    legend = :topleft,
    xticks = xticks,
    yticks = yticks,
    marker = :circle,
    # color = :Green,
    dpi = 600,
    # left_margin = mm, bottom_margin = 4mm,top_margin = 6mm,right_margin = 6mm
)


plt = plot!(
    Ns,
    Julia_times_rs4,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "Julia: Rodas4",
    marker = :circle,
    # color = :Green
)

plt = plot!(
    Ns,
    Julia_times_rs5p,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "Julia: Rodas5P",
    marker = :circle,
    # color = :Green
)

plt = plot!(
    Ns,
    Julia_EGArray_rs5p_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "Julia: Rodas5P (EnsembleGPUArray)",
    marker = :circle,
)


plt = plot!(
    Ns,
    Julia_EGArray_kv5_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "Julia: Kaverno5 (EnsembleGPUArray)",
    marker = :circle,
)



plt = plot!(
    Ns,
    JAX_times,
    xaxis = :log,
    yaxis = :log,
    linewidth = 2,
    label = "JAX: Kaverno5",
    marker = :ltriangle,
)


plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

isdir(plots_dir) || mkdir(plots_dir)


savefig(plt, joinpath(plots_dir, "Rober_adaptive_low_tol_$(Dates.value(Dates.now())).png"))
