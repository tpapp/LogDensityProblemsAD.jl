"""
Gradient AD implementation using Zygote.
"""
module LogDensityProblemsADZygoteExt

using LogDensityProblemsAD: ADGradientWrapper, logdensity

import ADTypes
import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
import Zygote

struct ZygoteGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

function ADgradient(::ADTypes.AutoZygote, ℓ; x::Union{Nothing,AbstractVector}=nothing)
    ZygoteGradientLogDensity(ℓ)
end

@deprecate ADgradient(::Val{:Zygote}, ℓ; x = nothing) ADgradient(ADTypes.AutoZygote(), ℓ; x)

Base.show(io::IO, ∇ℓ::ZygoteGradientLogDensity) = print(io, "Zygote AD wrapper for ", ∇ℓ.ℓ)

function logdensity_and_gradient(∇ℓ::ZygoteGradientLogDensity, x::AbstractVector)
    (; ℓ) = ∇ℓ
    y, back = Zygote.pullback(Base.Fix1(logdensity, ℓ), x)
    y, first(back(Zygote.sensitivity(y)))
end

end # module
