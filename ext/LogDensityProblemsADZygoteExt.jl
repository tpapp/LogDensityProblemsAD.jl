"""
Gradient AD implementation using Zygote.
"""
module LogDensityProblemsADZygoteExt

if isdefined(Base, :get_extension)
    using LogDensityProblemsAD: ADGradientWrapper, logdensity
    
    import LogDensityProblemsAD: ADgradient, logdensity_and_gradient    
    import Zygote
else
    using ..LogDensityProblemsAD: ADGradientWrapper, logdensity

    import ..LogDensityProblemsAD: ADgradient, logdensity_and_gradient
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
    (; ℓ) = ∇ℓ
    y, back = Zygote.pullback(Base.Fix1(logdensity, ℓ), x)
    y, first(back(Zygote.sensitivity(y)))
end

end # module
