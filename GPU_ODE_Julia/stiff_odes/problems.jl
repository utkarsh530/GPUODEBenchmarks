function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

function lorenz_jac(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    x = u[1]
    y = u[2]
    z = u[3]
    J11 = -σ
    J21 = ρ - z
    J31 = y
    J12 = σ
    J22 = -1
    J32 = x
    J13 = 0
    J23 = -x
    J33 = -β
    return SMatrix{3, 3}(J11, J21, J31, J12, J22, J32, J13, J23, J33)
end

function lorenz_tgrad(u, p, t)
    return SVector{3, eltype(u)}(0.0, 0.0, 0.0)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]

func = ODEFunction(lorenz, jac = lorenz_jac, tgrad = lorenz_tgrad)
lorenz_prob = ODEProblem{false}(func, u0, tspan, p)

function rober_f(internal_var___u, internal_var___p, t)
    internal_var___du1 = -(0.04f0) * internal_var___u[1] +
                         internal_var___p[1] * internal_var___u[2] *
                         internal_var___u[3]
    internal_var___du2 = (0.04f0 * internal_var___u[1] -
                          3.0f7 * internal_var___u[2]^2) -
                         internal_var___p[1] * internal_var___u[2] *
                         internal_var___u[3]
    internal_var___du3 = 3.0f7 * internal_var___u[2]^2
    return SVector{3,eltype(internal_var___u)}(internal_var___du1, internal_var___du2, internal_var___du3)
end

function rober_jac(internal_var___u, internal_var___p, t)
    internal_var___J11 = -(0.04f0)
    internal_var___J12 = internal_var___p[1] * internal_var___u[3]
    internal_var___J13 = internal_var___p[1] * internal_var___u[2]
    internal_var___J21 = 0.04f0 * 1
    internal_var___J22 = -2 * 3.0f7 * internal_var___u[2] -
                         internal_var___p[1] * internal_var___u[3]
    internal_var___J23 = -(internal_var___p[1]) * internal_var___u[2]
    internal_var___J31 = 0 * 1
    internal_var___J32 = 2 * 3.0f7 * internal_var___u[2]
    internal_var___J33 = 0 * 1
    return SMatrix{3, 3, eltype(internal_var___u)}(internal_var___J11, internal_var___J21, internal_var___J31,
                         internal_var___J12, internal_var___J22, internal_var___J32,
                         internal_var___J13, internal_var___J23, internal_var___J33)
end

function rober_tgrad(u, p, t)
    return SVector{3, eltype(u)}(0.0, 0.0, 0.0)
end

u0 = @SVector Float32[1.0, 0.0, 0.0]
p = @SVector Float32[1.0f4]

rober_prob = ODEProblem(ODEFunction(rober_f, jac = rober_jac, tgrad = rober_tgrad),
                        u0, (0.0f0, 1.0f5), p)
