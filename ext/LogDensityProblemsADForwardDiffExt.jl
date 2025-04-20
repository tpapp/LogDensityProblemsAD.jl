"""
Gradient AD implementation using ForwardDiff.
"""
module LogDensityProblemsADForwardDiffExt

using LogDensityProblemsAD: ADGradientWrapper, SIGNATURES, dimension, logdensity

import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
import ForwardDiff
import ForwardDiff: DiffResults

# Load DiffResults helpers
include("DiffResults_helpers.jl")

struct ForwardDiffLogDensity{L, C <: ForwardDiff.Chunk, T,
                             G <: Union{Nothing,ForwardDiff.GradientConfig}} <: ADGradientWrapper
    "supports zero-order evaluation `logdensity(ℓ, x)`"
    ℓ::L
    "chunk size for ForwardDiff"
    chunk::C
    "tag, or `nothing` for the default"
    tag::T
    "gradient config, or `nothing` if created for each evaluation"
    gradient_config::G
end

function Base.show(io::IO, ℓ::ForwardDiffLogDensity)
    print(io, "ForwardDiff AD wrapper for ", ℓ.ℓ,
          ", w/ chunk size ", ForwardDiff.chunksize(ℓ.chunk))
end

_ensure_chunk(chunk::ForwardDiff.Chunk) = chunk
_ensure_chunk(chunk::Integer) = ForwardDiff.Chunk(chunk)

_default_chunk(ℓ) = _ensure_chunk(dimension(ℓ))

function Base.copy(fℓ::ForwardDiffLogDensity{L,C,T,<:ForwardDiff.GradientConfig}) where {L,C,T}
    (; ℓ, chunk, tag, gradient_config) = fℓ
    ForwardDiffLogDensity(ℓ, chunk, tag, copy(gradient_config))
end

"""
$(SIGNATURES)

Make a `ForwardDiff.GradientConfig` for function `f` and input `x`. `tag = nothing` generates the default tag.
"""
function _make_gradient_config(f::F, x, chunk::ForwardDiff.Chunk, tag) where {F}
    gradient_config = if tag ≡ nothing
        ForwardDiff.GradientConfig(f, x, chunk)
    elseif tag isa ForwardDiff.Tag
        ForwardDiff.GradientConfig(f, x, chunk, tag)
    else
        ForwardDiff.GradientConfig(f, x, chunk, ForwardDiff.Tag(tag, eltype(x)))
    end
    gradient_config
end

"""
    ADgradient(:ForwardDiff, ℓ; chunk, tag, x)
    ADgradient(Val(:ForwardDiff), ℓ; chunk, tag, x)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keyword arguments:

- `chunk` can be used to set the chunk size, an integer or a `ForwardDiff.Chunk`

- `tag` (default: `nothing`) can be used to set a tag for `ForwardDiff`.
  If `tag` is neither `nothing` nor a `ForwardDiff.Tag`, the tag for `ForwardDiff` is set
  to `ForwardDiff.Tag(tag, eltype(x))` where `x` is the vector at which the logdensity and
  its gradient are evaluated. 

- `x` (default: `nothing`) will be used to preallocate a `ForwardDiff.GradientConfig` with
  the given vector. With the default, one is created for each evaluation.

Note that **pre-allocating a `ForwardDiff.GradientConfig` is not thread-safe**. You can
[`copy`](@ref) the results for concurrent evaluation:
```julia
∇ℓ1 = ADgradient(:ForwardDiff, ℓ; x = zeros(dimension(ℓ)))
∇ℓ2 = copy(∇ℓ1) # you can now use both, in different threads
```

See also the ForwardDiff documentation regarding
[`ForwardDiff.GradientConfig`](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#Preallocating/Configuring-Work-Buffers)
and [chunks and tags](https://juliadiff.org/ForwardDiff.jl/stable/user/advanced/).
"""
function ADgradient(::Val{:ForwardDiff}, ℓ;
                    chunk::Union{Integer,ForwardDiff.Chunk} = _default_chunk(ℓ),
                    tag = nothing,
                    x::Union{Nothing,AbstractVector} = nothing)
    _chunk = _ensure_chunk(chunk)
    gradient_config = if x ≡ nothing
        nothing
    else
        _make_gradient_config(Base.Fix1(logdensity, ℓ), x, _chunk, tag)
    end
    ForwardDiffLogDensity(ℓ, _chunk, tag, gradient_config)
end

function logdensity_and_gradient(fℓ::ForwardDiffLogDensity, x::AbstractVector)
    (; ℓ, chunk, tag, gradient_config) = fℓ
    buffer = _diffresults_buffer(x)
    ℓ′ = Base.Fix1(logdensity, ℓ)
    if gradient_config ≡ nothing
        gradient_config = _make_gradient_config(ℓ′, x, chunk, tag)
    end
    result = ForwardDiff.gradient!(buffer, ℓ′, x, gradient_config)
    _diffresults_extract(result)
end

end # module
