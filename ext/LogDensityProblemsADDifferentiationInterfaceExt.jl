module LogDensityProblemsADDifferentiationInterfaceExt

using DocStringExtensions: FIELDS, TYPEDEF
import LogDensityProblems, LogDensityProblemsAD
using ADTypes: AbstractADType
import DifferentiationInterface as DI

"""
$(TYPEDEF)

Gradient wrapper which uses [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)

# Fields

$(FIELDS)
"""
struct DIGradient{B<:AbstractADType,P,L} <: LogDensityProblemsAD.ADGradientWrapper
    """
    one of the autodiff backend types defined in
    [ADTypes.jl](https://github.com/SciML/ADTypes.jl), for example `ADTypes.AutoForwardDiff()`
    """
    backend::B
    """
    either `nothing` or the output of `DifferentiationInterface.prepare_gradient`
    applied to the logdensity and the provided input
    """
    prep::P
    """
    logdensity function, which supports the `LogDensityProblems` interface (with at
    least `LogDensityOrder{0}`, `backend` is used for gradient)
    """
    ℓ::L
end

@inline _logdensity_callable(ℓ) = Base.Fix1(LogDensityProblems.logdensity, ℓ)

function LogDensityProblemsAD.ADgradient(backend::AbstractADType, ℓ;
                                         x::LogDensityProblemsAD.__VALIDX = nothing)
    if x === nothing
        prep = nothing
    else
        prep = DI.prepare_gradient(_logdensity_callable(ℓ), backend, x)
    end
    return DIGradient(backend, prep, ℓ)
end

function LogDensityProblemsAD.logdensity_and_gradient(∇ℓ::DIGradient, x::AbstractVector)
    (; backend, prep, ℓ) = ∇ℓ
    if prep ≡ nothing
        DI.value_and_gradient(_logdensity_callable(ℓ), backend, x)
    else
        DI.value_and_gradient(_logdensity_callable(ℓ), prep, backend, x)
    end
end

end
