#####
##### Gradient AD implementation using ForwardDiff
#####

import .ForwardDiff

import .ForwardDiff.DiffResults # should load DiffResults_helpers.jl

struct ForwardDiffLogDensity{L, C} <: ADGradientWrapper
    ℓ::L
    gradientconfig::C
end

function Base.show(io::IO, ℓ::ForwardDiffLogDensity)
    print(io, "ForwardDiff AD wrapper for ", ℓ.ℓ,
          ", w/ chunk size ", length(ℓ.gradientconfig.seeds))
end

_chunk(chunk::ForwardDiff.Chunk) = chunk
_chunk(chunk::Integer) = ForwardDiff.Chunk(chunk)

_default_chunk(ℓ) = _chunk(dimension(ℓ))

_default_gradientconfig(ℓ, chunk, ::Nothing) = _default_gradientconfig(ℓ, chunk, zeros(dimension(ℓ)))
function _default_gradientconfig(ℓ, chunk, x::AbstractVector)
    return ForwardDiff.GradientConfig(Base.Fix1(logdensity, ℓ), x, _chunk(chunk))
end

"""
    ADgradient(:ForwardDiff, ℓ; input, chunk, gradientconfig)
    ADgradient(Val(:ForwardDiff), ℓ; input, chunk, gradientconfig)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keywords are passed on to `ForwardDiff.GradientConfig` to customize the setup. In
particular, chunk size can be set with a `chunk` keyword argument (accepting an integer or a
`ForwardDiff.Chunk`), and the underlying vector used by `ForwardDiff` can be set with the
`x` keyword argument (accepting an `AbstractVector`).
"""
function ADgradient(::Val{:ForwardDiff}, ℓ;
                    x = nothing,
                    chunk = _default_chunk(ℓ),
                    gradientconfig = _default_gradientconfig(ℓ, chunk, x))
    ForwardDiffLogDensity(ℓ, gradientconfig)
end

function logdensity_and_gradient(fℓ::ForwardDiffLogDensity, x::AbstractVector)
    @unpack ℓ, gradientconfig = fℓ
    buffer = _diffresults_buffer(x)
    result = ForwardDiff.gradient!(buffer, Base.Fix1(logdensity, ℓ), x, gradientconfig)
    _diffresults_extract(result)
end
