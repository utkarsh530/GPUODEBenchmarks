using CUDA, DiffEqGPU, OrdinaryDiffEq, Plots, Serialization, StaticArrays, Distributions, LinearAlgebra
import DataInterpolations
const DI = DataInterpolations

trajectories = 100
u0 = @SVector [0.0f0, 0.0f0, 10000.0f0, 0f0, 0f0, 0f0]
tspan = (0.0f0, 50.0f0)
saveat = LinRange(tspan..., 100)
p = @SVector [25f0, 225f0, 9.807f0]
CdS_dist = Normal(0f0, 1f0)

## Example where interpolation is performed on GPU

data = deserialize("forecast.txt")
N = length(data.altitude)

weather_sa = map(data.altitude, data.windx, data.windy, data.density) do alt, wx, wy, ρ
    SVector{4}(alt, wx, wy, ρ)
end

data = deserialize("forecast.txt")
N = length(data.altitude)

weather_sa = map(data.altitude, data.windx, data.windy, data.density) do alt, wx, wy, ρ
    SVector{4}(alt, wx, wy, ρ)
end

weather_sa = SVector{length(weather_sa)}(weather_sa)

interp = DI.LinearInterpolation{true}(hcat(weather_sa...),data.altitude)

function get_weather(itp::DI.LinearInterpolation, z)
    weather = itp(z)
    wind = SVector{3}(weather[2], weather[3], 0f0)
    ρ = weather[4]
    wind, ρ
end


### solving the ODE on GPU + Interpolation using DataInterpolations

function ballistic_gpu(u, p, t)
    CdS, mass, g = p[1]
    interp = p[2]
    vel = @view u[4:6]

    wind, ρ = get_weather(interp, u[3])

    airvelocity = vel - wind
    airspeed = norm(airvelocity)
    accel = -(ρ * CdS * airspeed) / (2 * mass) * airvelocity - mass*SVector{3}(0f0, 0f0, g)

    return SVector{6}(vel..., accel...)
end


prob_interp = ODEProblem{false}(ballistic_gpu, u0, tspan, (p, interp))

prob_func = (prob, i, repeat) -> remake(prob_interp, p = (p + SVector{3}(rand(CdS_dist), 0f0, 0f0), interp))
eprob_interp = EnsembleProblem(prob_interp, prob_func = prob_func, safetycopy = false)

esol_gpu = solve(eprob_interp, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0); trajectories, saveat)

using BenchmarkTools

@benchmark esol_gpu = solve(eprob_interp, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0); trajectories, saveat)


## Replace interpolation with textured-memory


weather = map(weather_sa) do w
    (w...,)
end

weather_TA = CuTextureArray(weather)
texture = CuTexture(weather_TA; address_mode = CUDA.ADDRESS_MODE_CLAMP, normalized_coordinates = true, interpolation = CUDA.LinearInterpolation())

## Test Texture interpolation
idx = LinRange(0f0, 1f0, 4000)
idx_gpu = CuArray(idx)
idx_tlu = (1f0-1f0/N)*idx_gpu .+ 0.5f0/N  # normalized table lookup form https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-lookup
dst_gpu = CuArray{NTuple{4, Float32}}(undef, size(idx))
dst_gpu .= getindex.(Ref(texture), idx_tlu)  # interpolation ℝ->ℝ⁴


def_zmax = data.altitude[end]
N = length(data.altitude)
@inline function get_weather(tex, z, zmax, N)
    idx = (1f0-1f0/N)*z/zmax + 0.5f0/N # normalized input for table lookup based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-lookup
    weather = tex[idx]
    wind = SVector{3}(weather[2], weather[3], 0f0)
    ρ = weather[4]
    wind, ρ
end

### Experimentation

function ballistic_t(u, p, t)
    CdS, mass, g = p[1]
    interp = p[2]
    zmax = p[3]
    N = p[4]
    vel = @view u[4:6]

    wind, ρ = get_weather(interp, u[3], zmax, N)

    airvelocity = vel - wind
    airspeed = norm(airvelocity)
    accel = -(ρ * CdS * airspeed) / (2 * mass) * airvelocity - mass*SVector{3}(0f0, 0f0, g)

    return SVector{6}(vel..., accel...)
end

prob_tx = ODEProblem(ballistic_t, u0, tspan, (p, texture, def_zmax, N))

using Adapt

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuArray{<:ODEProblem})
    # first convert the contained ODE problems
    y = CuArray(adapt.(Ref(to), Array(x)))
    # continue doing what the default method does
    Base.unsafe_convert(CuDeviceArray{eltype(y),ndims(y),CUDA.AS.Global}, y)
end

prob_func = (prob, i, repeat) -> remake(prob, p = (p + SVector{3}(rand(CdS_dist), 0f0, 0f0), texture, def_zmax, N))
eprob_texture = EnsembleProblem(prob_tx, prob_func = prob_func, safetycopy = false)

esol_gpu = solve(eprob_texture, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0); trajectories, saveat)

@benchmark esol_gpu = solve(eprob_texture, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0); trajectories, saveat)
