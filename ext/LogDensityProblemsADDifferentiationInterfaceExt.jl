module LogDensityProblemsADDifferentiationInterfaceExt

import LogDensityProblemsAD
using ADTypes: AbstractADType
import DifferentiationInterface as DI

"""
    DIGradient <: LogDensityProblemsAD.ADGradientWrapper

Gradient wrapper which uses [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)

# Fields

- `backend::AbstractADType`: one of the autodiff backend types defined in [ADTypes.jl](https://github.com/SciML/ADTypes.jl), for example `ADTypes.AutoForwardDiff()`
- `prep`: either `nothing` or the output of `DifferentiationInterface.prepare_gradient` applied to the logdensity and the provided input
- `ℓ`: logdensity function, amenable to `LogDensityProblemsAD.logdensity(ℓ, x)`
"""
struct DIGradient{B<:AbstractADType,P,L} <: LogDensityProblemsAD.ADGradientWrapper
    backend::B
    prep::P
    ℓ::L
end

function logdensity_switched(x, ℓ)
    # active argument must come first in DI
    return LogDensityProblemsAD.logdensity(ℓ, x)
end

function LogDensityProblemsAD.ADgradient(backend::AbstractADType, ℓ;
                                         x::LogDensityProblemsAD.__VALIDX = nothing)
    if x === nothing
        prep = nothing
    else
        prep = DI.prepare_gradient(logdensity_switched, backend, x, DI.Constant(ℓ))
    end
    return DIGradient(backend, prep, ℓ)
end

function LogDensityProblemsAD.logdensity_and_gradient(∇ℓ::DIGradient, x::AbstractVector)
    (; backend, prep, ℓ) = ∇ℓ
    if prep === nothing
        return DI.value_and_gradient(logdensity_switched, backend, x, DI.Constant(ℓ))
    else
        return DI.value_and_gradient(logdensity_switched, prep, backend, x, DI.Constant(ℓ))
    end
end

end
