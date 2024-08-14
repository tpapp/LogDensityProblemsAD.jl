"""
Gradient AD implementation using Tracker.
"""
module LogDensityProblemsADTrackerExt

if isdefined(Base, :get_extension)
    using LogDensityProblemsAD: ADGradientWrapper, logdensity

    import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
    import Tracker
else
    using ..LogDensityProblemsAD: ADGradientWrapper, logdensity

    import ..LogDensityProblemsAD: ADgradient, logdensity_and_gradient
    import ..Tracker
end

struct TrackerGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

"""
    ADgradient(:Tracker, ℓ)
    ADgradient(Val(:Tracker), ℓ)

Gradient using algorithmic/automatic differentiation via Tracker.

This package has been deprecated in favor of Zygote, but we keep the interface available.
"""
ADgradient(::Val{:Tracker}, ℓ) = TrackerGradientLogDensity(ℓ)

Base.show(io::IO, ∇ℓ::TrackerGradientLogDensity) = print(io, "Tracker AD wrapper for ", ∇ℓ.ℓ)

function logdensity_and_gradient(∇ℓ::TrackerGradientLogDensity, x::AbstractVector{T}) where {T}
    (; ℓ) = ∇ℓ
    y, back = Tracker.forward(x -> logdensity(ℓ, x), x)
    yval = Tracker.data(y)
    # work around https://github.com/FluxML/Flux.jl/issues/497
    z = T <: Real ? zero(T) : 0.0
    S = typeof(z + 0.0)
    S(yval)::S, (S.(first(Tracker.data.(back(1)))))::Vector{S}
end

end # module
