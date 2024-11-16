"""
Automatic differentiation backends for LogDensityProblems.
"""
module LogDensityProblemsAD

export ADgradient

using DocStringExtensions: SIGNATURES
import LogDensityProblems: logdensity, logdensity_and_gradient, capabilities, dimension
using LogDensityProblems: LogDensityOrder

#####
##### AD wrappers --- interface and generic code
#####

"""
An abstract type that wraps another log density for calculating the gradient via AD.

Automatically defines the methods `capabilities`, `dimension`, and `logdensity` forwarding
to the field `ℓ`, subtypes should define a [`logdensity_and_gradientent`](@ref).

This is an implementation helper, not part of the API.
"""
abstract type ADGradientWrapper end

logdensity(ℓ::ADGradientWrapper, x::AbstractVector) = logdensity(ℓ.ℓ, x)

capabilities(::Type{<:ADGradientWrapper}) = LogDensityOrder{1}()

dimension(ℓ::ADGradientWrapper) = dimension(ℓ.ℓ)

Base.parent(ℓ::ADGradientWrapper) = ℓ.ℓ

Base.copy(x::ADGradientWrapper) = x # no-op, except for ForwardDiff

"""
$(SIGNATURES)

Wrap `P` using automatic differentiation to obtain a gradient.

`kind` is usually a `Val` type with a symbol that refers to a package, for example
```julia
ADgradient(Val(:ForwardDiff), P)
ADgradient(Val(:ReverseDiff), P)
ADgradient(Val(:Zygote), P)
```
Some methods may be loaded only conditionally after the relevant package is loaded (eg
`using Zygote`).

The symbol can also be used directly as eg

```julia
ADgradient(:ForwardDiff, P)
```

and should mostly be equivalent if the compiler manages to fold the constant.

The function `parent` can be used to retrieve the original argument.

!!! note
    With the default options, automatic differentiation preserves thread-safety. See
    exceptions and workarounds in the docstring for each backend.
"""
ADgradient(kind::Symbol, P; kwargs...) = ADgradient(Val{kind}(), P; kwargs...)

# Better error message if users forget to load the AD package
if isdefined(Base.Experimental, :register_error_hint)
    _extract_val(::Val{T}) where {T} = T
    function __init__()
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            if exc.f === ADgradient && length(argtypes) == 2 && first(argtypes) <: Val
                kind = _extract_val(first(argtypes))
                print(io, "\nDon't know how to AD with $(kind), consider `import $(kind)` if there is such a package.")
            end
        end
    end
end

#####
##### Empty method definitions for easier discoverability and backward compatibility
#####
function benchmark_ForwardDiff_chunks end
function heuristic_chunks end

end # module
