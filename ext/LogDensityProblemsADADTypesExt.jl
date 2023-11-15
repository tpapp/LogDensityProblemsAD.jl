module LogDensityProblemsADADTypesExt

if isdefined(Base, :get_extension)
    import LogDensityProblemsAD
    import ADTypes
else
    import ..LogDensityProblemsAD
    import ..ADTypes
end

"""
    ADgradient(ad::ADTypes.AbstractADType, ℓ)

Wrap log density `ℓ` using automatic differentiation (AD) of type `ad` to obtain a gradient.

Currently,
- `ad::ADTypes.AutoEnzyme`
- `ad::ADTypes.AutoForwardDiff`
- `ad::ADTypes.AutoReverseDiff`
- `ad::ADTypes.AutoTracker`
- `ad::ADTypes.AutoZygote`
are supported.
The AD configuration specified by `ad` is forwarded to the corresponding calls of `ADgradient(Val(...), ℓ)`.    
"""
LogDensityProblemsAD.ADgradient(::ADTypes.AbstractADType, ℓ)

function LogDensityProblemsAD.ADgradient(::ADTypes.AutoEnzyme, ℓ)
    return LogDensityProblemsAD.ADgradient(Val(:Enzyme), ℓ)
end

function LogDensityProblemsAD.ADgradient(::ADTypes.AutoForwardDiff{C}, ℓ) where {C}
    if C === nothing
        # Use default chunk size
        return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ)
    else
        return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; chunk=C)
    end
end

function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoReverseDiff, ℓ; x::Union{Nothing,AbstractVector}=nothing)
    return LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), ℓ; compile=Val(ad.compile), x=x)
end

function LogDensityProblemsAD.ADgradient(::ADTypes.AutoTracker, ℓ)
    return LogDensityProblemsAD.ADgradient(Val(:Tracker), ℓ)
end


function LogDensityProblemsAD.ADgradient(::ADTypes.AutoZygote, ℓ)
    return LogDensityProblemsAD.ADgradient(Val(:Zygote), ℓ)
end

end # module
