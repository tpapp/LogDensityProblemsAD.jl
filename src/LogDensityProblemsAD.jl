"""
Automatic differentiation backends for LogDensityProblems.
"""
module LogDensityProblemsAD

export ADgradient

using DocStringExtensions: FUNCTIONNAME
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
$(FUNCTIONNAME)(backend, ℓ; x = nothing)

Wrap `ℓ` using automatic differentiation to obtain a gradient. `ℓ` should support the
`LogDensityProblems` API for calculating log densities (gradient not needed).

`backend` is a backend defined in [ADTypes.jl](https://docs.sciml.ai/ADTypes/stable/),
eg `AutoForwardDiff()`.

Some methods may be loaded only conditionally after the relevant package is loaded (eg
`using Mooncake`).

The function `parent` can be used to retrieve `ℓ`.

`x` can be provided is a vector to “prepare” the gradient. This may result in faster
runtimes. When this is not applicable, this argument is silently ignored.

!!! note
    With the default options, automatic differentiation preserves thread-safety. See
    exceptions and workarounds in the docstring for each backend.
"""
function ADgradient end

ADgradient(kind::Symbol, P; kwargs...) = ADgradient(Val{kind}(), P; kwargs...)

#####
##### Empty method definitions for easier discoverability and backward compatibility
#####

function benchmark_ForwardDiff_chunks end
function heuristic_chunks end

end # module
