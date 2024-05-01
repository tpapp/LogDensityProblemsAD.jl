module LogDensityProblemsADDifferentiationInterfaceExt

using LogDensityProblemsAD: ADGradientWrapper, LogDensityProblemsAD, logdensity
using DifferentiationInterface: GradientExtras, prepare_gradient, value_and_gradient
using DifferentiationInterface.ADTypes: AbstractADType

struct DIGradient{B,L,E<:Union{Nothing,GradientExtras}} <: ADGradientWrapper
    backend::B
    ℓ::L
    extras::E
end

function Base.show(io::IO, ∇ℓ::DIGradient)
    print(io, "DifferentiationInterface AD wrapper for $(∇ℓ.ℓ) with backend $(∇ℓ.backend)")
end

Base.copy(∇ℓ::DIGradient) = deepcopy(∇ℓ)

"""
    ADgradient(ad::ADTypes.AbstractADType, ℓ; x=nothing)

Wrap log density `ℓ` using automatic differentiation (AD) of type `ad` to obtain a gradient.

If `x` is provided, prepare the gradient calls by building a config / tape / etc.
"""
function LogDensityProblemsAD.ADgradient(backend::AbstractADType, ℓ; x=nothing)
    if isnothing(x)
        return DIGradient(backend, ℓ, nothing)
    else
        extras = prepare_gradient(Base.Fix1(logdensity, ℓ), backend, x)
        return DIGradient(backend, ℓ, extras)
    end
end

function LogDensityProblemsAD.logdensity_and_gradient(∇ℓ::DIGradient, x::AbstractVector)
    (; backend, ℓ, extras) = ∇ℓ
    if isnothing(extras)
        return value_and_gradient(Base.Fix1(logdensity, ℓ), backend, x)
    else
        return value_and_gradient(Base.Fix1(logdensity, ℓ), backend, x, extras)
    end
end

end # module
