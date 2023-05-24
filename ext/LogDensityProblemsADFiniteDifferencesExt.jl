"""
Gradient implementation using FiniteDifferences.
"""
module LogDensityProblemsADFiniteDifferencesExt

if isdefined(Base, :get_extension)
    using LogDensityProblemsAD: ADGradientWrapper, logdensity
    using LogDensityProblemsAD.SimpleUnPack: @unpack

    import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
    import FiniteDifferences
else
    using ..LogDensityProblemsAD: ADGradientWrapper, logdensity
    using ..LogDensityProblemsAD.SimpleUnPack: @unpack

    import ..LogDensityProblemsAD: ADgradient, logdensity_and_gradient
    import ..FiniteDifferences
end

struct FiniteDifferencesGradientLogDensity{L,M} <: ADGradientWrapper
    ℓ::L
    "finite difference method"
    fdm::M
end

"""
    ADgradient(:FiniteDifferences, ℓ; fdm = central_fdm(5, 1))
    ADgradient(Val(:FiniteDifferences), ℓ; fdm = central_fdm(5, 1))

Gradient using FiniteDifferences, mainly intended for checking results from other algorithms.

# Keyword arguments

- `fdm`: the finite difference method. Defaults to `central_fdm(5, 1)`.
"""
function ADgradient(::Val{:FiniteDifferences}, ℓ; fdm = FiniteDifferences.central_fdm(5, 1))
    FiniteDifferencesGradientLogDensity(ℓ, fdm)
end

function Base.show(io::IO, ∇ℓ::FiniteDifferencesGradientLogDensity)
    print(io, "FiniteDifferences AD wrapper for ", ∇ℓ.ℓ, " with ", ∇ℓ.fdm)
end

function logdensity_and_gradient(∇ℓ::FiniteDifferencesGradientLogDensity, x::AbstractVector)
    @unpack ℓ, fdm = ∇ℓ
    y = logdensity(ℓ, x)
    ∇y = only(FiniteDifferences.grad(fdm, Base.Fix1(logdensity, ℓ), x))
    y, ∇y
end

end # module
