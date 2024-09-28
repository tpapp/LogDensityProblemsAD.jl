module LogDensityProblemsADDifferentiationInterfaceExt

if isdefined(Base, :get_extension)
    import LogDensityProblemsAD
    import ADTypes
    import DifferentiationInterface as DI
else
    import ..LogDensityProblemsAD
    import ..ADTypes
    import ..DifferentiationInterface as DI
end

struct DIGradient{B,P,L} <: LogDensityProblemsAD.ADGradientWrapper
    backend::B
    prep::P
    ℓ::L
end

function logdensity_switched(x, ℓ)
    # active argument must come first in DI
    return LogDensityProblemsAD.logdensity(ℓ, x)
end

function LogDensityProblemsAD.ADgradient(backend::ADTypes.AbstractADType, ℓ; x=nothing)
    if isnothing(x)
        prep = nothing
    else
        prep = DI.prepare_gradient(logdensity_switched, backend, x, DI.Constant(ℓ))
    end
    return DIGradient(backend, prep, ℓ)
end

function LogDensityProblemsAD.logdensity_and_gradient(∇ℓ::DIGradient, x)
    backend, prep, ℓ = ∇ℓ.backend, ∇ℓ.prep, ∇ℓ.ℓ
    if isnothing(prep)
        return DI.value_and_gradient(logdensity_switched, backend, x, DI.Constant(ℓ))
    else
        return DI.value_and_gradient(logdensity_switched, prep, backend, x, DI.Constant(ℓ))
    end
end

end
