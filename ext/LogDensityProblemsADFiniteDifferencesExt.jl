"""
Gradient implementation using FiniteDifferences.
"""
module LogDensityProblemsADFiniteDifferencesExt

using LogDensityProblemsAD: ADGradientWrapper, logdensity

using ADTypes: AutoFiniteDifferences
import LogDensityProblemsAD: ADgradient, logdensity_and_gradient, __VALIDX
import FiniteDifferences

struct FiniteDifferencesGradientLogDensity{L,M} <: ADGradientWrapper
    ℓ::L
    "finite difference method"
    fdm::M
end

function ADgradient(ad::AutoFiniteDifferences, ℓ; x::__VALIDX = nothing)
    FiniteDifferencesGradientLogDensity(ℓ, ad.fdm)
end

@deprecate(ADgradient(::Val{:FiniteDifferences}, ℓ; fdm = FiniteDifferences.central_fdm(5, 1)),
           ADgradient(AutoFiniteDifferences(; fdm), ℓ))

function Base.show(io::IO, ∇ℓ::FiniteDifferencesGradientLogDensity)
    print(io, "FiniteDifferences AD wrapper for ", ∇ℓ.ℓ, " with ", ∇ℓ.fdm)
end

function logdensity_and_gradient(∇ℓ::FiniteDifferencesGradientLogDensity,
                                 x::AbstractVector{T}) where T
    (; ℓ, fdm) = ∇ℓ
    y = logdensity(ℓ, x)
    S = float(T)
    ∇y = only(FiniteDifferences.grad(fdm, Base.Fix1(logdensity, ℓ), x))
    y, convert(Vector{S}, ∇y)::Vector{S}
end

end # module
