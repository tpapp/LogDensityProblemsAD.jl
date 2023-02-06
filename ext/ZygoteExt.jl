"""
Gradient AD implementation using Zygote.
"""
module ZygoteExt

using LogDensityProblemsAD: ADGradientWrapper, EXTENSIONS_SUPPORTED, logdensity
using LogDensityProblemsAD.UnPack: @unpack

import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
if EXTENSIONS_SUPPORTED
    import Zygote
else
    import ..Zygote
end

struct ZygoteGradientLogDensity{L} <: ADGradientWrapper
    ℓ::L
end

"""
    ADgradient(:Zygote, ℓ)
    ADgradient(Val(:Zygote), ℓ)

Gradient using algorithmic/automatic differentiation via Zygote.
"""
ADgradient(::Val{:Zygote}, ℓ) = ZygoteGradientLogDensity(ℓ)

Base.show(io::IO, ∇ℓ::ZygoteGradientLogDensity) = print(io, "Zygote AD wrapper for ", ∇ℓ.ℓ)

function logdensity_and_gradient(∇ℓ::ZygoteGradientLogDensity, x::AbstractVector)
    @unpack ℓ = ∇ℓ
    y, back = Zygote.pullback(Base.Fix1(logdensity, ℓ), x)
    y, first(back(Zygote.sensitivity(y)))
end

end # module
