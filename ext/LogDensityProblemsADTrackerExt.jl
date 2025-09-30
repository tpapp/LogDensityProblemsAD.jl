"""
Gradient AD implementation using Tracker.
"""
module LogDensityProblemsADTrackerExt

using LogDensityProblemsAD: ADGradientWrapper, logdensity

using ADTypes: AutoTracker
import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
import Tracker

struct TrackerGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

ADgradient(::AutoTracker, ℓ; x = nothing) = TrackerGradientLogDensity(ℓ)

@deprecate ADgradient(::Val{:Tracker}, ℓ; x = nothing) ADgradient(AutoTracker(), ℓ; x)

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
