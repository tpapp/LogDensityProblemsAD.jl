"""
Gradient AD implementation using ReverseDiff.
"""
module LogDensityProblemsADReverseDiffExt

using LogDensityProblemsAD: ADGradientWrapper, dimension, logdensity

using ADTypes: AutoReverseDiff
import LogDensityProblemsAD: ADgradient, logdensity_and_gradient, __VALIDX
import ReverseDiff
import ReverseDiff: DiffResults

# Load DiffResults helpers
include("DiffResults_helpers.jl")

struct ReverseDiffLogDensity{L,C} <: ADGradientWrapper
    ℓ::L
    compiledtape::C
end

@deprecate(ADgradient(::Val{:ReverseDiff}, ℓ; compile = Val(false)),
           ADgradient(AutoReverseDiff(; compile), ℓ))

_compiledtape(ℓ, ::Val{false}, x) = nothing
_compiledtape(ℓ, ::Val{true}, ::Nothing) = _compiledtape(ℓ, Val(true), zeros(dimension(ℓ)))
function _compiledtape(ℓ, ::Val{true}, x)
    tape = ReverseDiff.GradientTape(Base.Fix1(logdensity, ℓ), x)
    return ReverseDiff.compile(tape)
end

function ADgradient(::AutoReverseDiff{C}, ℓ;
                    x::__VALIDX = nothing) where C
    ReverseDiffLogDensity(ℓ, _compiledtape(ℓ, Val(C), x))
end

function Base.show(io::IO, ∇ℓ::ReverseDiffLogDensity)
    print(io, "ReverseDiff AD wrapper for ", ∇ℓ.ℓ, " (")
    if ∇ℓ.compiledtape === nothing
        print(io, "no ")
    end
    print(io, "compiled tape)")
end

function logdensity_and_gradient(∇ℓ::ReverseDiffLogDensity, x::AbstractVector)
    (; ℓ, compiledtape) = ∇ℓ
    buffer = _diffresults_buffer(x)
    if compiledtape === nothing
        result = ReverseDiff.gradient!(buffer, Base.Fix1(logdensity, ℓ), x)
    else
        result = ReverseDiff.gradient!(buffer, compiledtape, x)
    end
    _diffresults_extract(result)
end

end # module
