"""
Automatic differentiation backends for LogDensityProblems.
"""
module LogDensityProblemsAD

export ADgradient

using DocStringExtensions: SIGNATURES
import LogDensityProblems: logdensity, logdensity_and_gradient, capabilities, dimension
using LogDensityProblems: LogDensityOrder

import UnPack

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

function ADgradient(v::Val{kind}, P; kwargs...) where kind
    @info "Don't know how to AD with $(kind), consider `import $(kind)` if there is such a package."
    throw(MethodError(ADgradient, (v, P)))
end

#####
##### Empty method definitions for easier discoverability and backward compatibility
#####
function benchmark_ForwardDiff_chunks end
function heuristic_chunks end

# Backward compatible AD wrappers on Julia versions that do not support extensions
# TODO: Replace with proper version
const EXTENSIONS_SUPPORTED = isdefined(Base, :get_extension)
if !EXTENSIONS_SUPPORTED
    using Requires: @require
end
function __init__()
    @static if !EXTENSIONS_SUPPORTED
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/ForwardDiffExt.jl")
            @require BenchmarkTools="6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf" begin
                include("../ext/ForwardDiffBenchmarkToolsExt.jl")
            end
        end
        @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("../ext/TrackerExt.jl")
        @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" include("../ext/ZygoteExt.jl")
        @require ReverseDiff="37e2e3b7-166d-5795-8a7a-e32c996b4267" include("../ext/ReverseDiffExt.jl")
        @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" include("../ext/EnzymeExt.jl")
    end
end

end # module
