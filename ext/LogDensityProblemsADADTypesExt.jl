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

Passing `x` as a keyword argument means that the gradient operator will be "prepared" for the specific type and size of the array `x`. This can speed up further evaluations on similar inputs, but will likely cause errors if the new inputs have a different type or size. With ReverseDiff, it can also yield incorrect results if the logdensity contains value-dependent control flow.

!!! warning
    If you want to use another backend from ADTypes which is not in the list above, or if you want to provide `x` for preparation, you need to load [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl) first.
    See the documentation of that package, especially `DifferentiationInterface.prepare_gradient`, for more details on preparation.
"""
LogDensityProblemsAD.ADgradient(::ADTypes.AbstractADType, ℓ)

function LogDensityProblemsAD.ADgradient(::ADTypes.AutoEnzyme, ℓ)
    return LogDensityProblemsAD.ADgradient(Val(:Enzyme), ℓ)
end

function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoForwardDiff{C}, ℓ) where {C}
    if C === nothing
        # Use default chunk size
        return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; tag = ad.tag)
    else
        return LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; chunk = C, tag = ad.tag)
    end
end

function LogDensityProblemsAD.ADgradient(ad::ADTypes.AutoReverseDiff{T}, ℓ) where {T}
    return LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), ℓ; compile = Val(T))
end

function LogDensityProblemsAD.ADgradient(::ADTypes.AutoTracker, ℓ)
    return LogDensityProblemsAD.ADgradient(Val(:Tracker), ℓ)
end


function LogDensityProblemsAD.ADgradient(::ADTypes.AutoZygote, ℓ)
    return LogDensityProblemsAD.ADgradient(Val(:Zygote), ℓ)
end

end # module
