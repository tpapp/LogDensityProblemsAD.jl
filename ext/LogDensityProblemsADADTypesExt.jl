module LogDensityProblemsADADTypesExt

if isdefined(Base, :get_extension)
    import LogDensityProblemsAD
    import ADTypes
else
    import ..LogDensityProblemsAD
    import ..ADTypes
end

"""
    ADgradient(ad::ADTypes.AbstractADType, ℓ; x=nothing)

Wrap log density `ℓ` using automatic differentiation (AD) of type `ad` to obtain a gradient.

Currently,
- `ad::ADTypes.AutoEnzyme`
- `ad::ADTypes.AutoForwardDiff`
- `ad::ADTypes.AutoReverseDiff`
- `ad::ADTypes.AutoTracker`
- `ad::ADTypes.AutoZygote`
are supported with custom implementations.
The AD configuration specified by `ad` is forwarded to the corresponding calls of `ADgradient(Val(...), ℓ)`.

Passing `x` as a keyword argument means that the gradient operator will be "prepared" for the specific type and size of the array `x`. This can speed up further evaluations on similar inputs, but will likely cause errors if the new inputs have a different type or size. With `AutoReverseDiff`, it can also yield incorrect results if the logdensity contains value-dependent control flow.

If you want to use another backend from [ADTypes.jl](https://github.com/SciML/ADTypes.jl) which is not in the list above, you need to load [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl) first.
"""
LogDensityProblemsAD.ADgradient(::ADTypes.AbstractADType, ℓ)

function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoEnzyme, ℓ; x::Union{Nothing,AbstractVector}=nothing)
    if ad.mode === nothing
        # Use default mode (Enzyme.Reverse)
        return LogDensityProblemsAD.ADgradient(Val(:Enzyme), ℓ)
    else
        return LogDensityProblemsAD.ADgradient(Val(:Enzyme), ℓ; mode=ad.mode)
    end
end

function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoForwardDiff{C}, ℓ; x::Union{Nothing,AbstractVector}=nothing) where {C}
    if C === nothing
        # Use default chunk size
        return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; tag = ad.tag, x=x)
    else
        return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; chunk = C, tag = ad.tag, x=x)
    end
end

function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoReverseDiff{T}, ℓ; x::Union{Nothing,AbstractVector}=nothing) where {T}
    return LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), ℓ; compile = Val(T), x=x)
end

function LogDensityProblemsAD.ADgradient(::ADTypes.AutoTracker, ℓ; x::Union{Nothing,AbstractVector}=nothing)
    return LogDensityProblemsAD.ADgradient(Val(:Tracker), ℓ)
end


function LogDensityProblemsAD.ADgradient(::ADTypes.AutoZygote, ℓ; x::Union{Nothing,AbstractVector}=nothing)
    return LogDensityProblemsAD.ADgradient(Val(:Zygote), ℓ)
end

end # module
