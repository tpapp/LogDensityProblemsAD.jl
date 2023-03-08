"""
Gradient AD implementation using ForwardDiff.
"""
module LogDensityProblemsADForwardDiffExt

using LogDensityProblemsAD: ADGradientWrapper, EXTENSIONS_SUPPORTED, SIGNATURES, dimension, logdensity
using LogDensityProblemsAD.UnPack: @unpack

import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
if EXTENSIONS_SUPPORTED
    import ForwardDiff
    import ForwardDiff: DiffResults
else
    import ..ForwardDiff
    import ..ForwardDiff: DiffResults
end

# Load DiffResults helpers
include("DiffResults_helpers.jl")

struct ForwardDiffLogDensity{L, C <: ForwardDiff.Chunk, T <: Union{Nothing,ForwardDiff.Tag},
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

_chunk(chunk::ForwardDiff.Chunk) = chunk
_chunk(chunk::Integer) = ForwardDiff.Chunk(chunk)

_default_chunk(ℓ) = _chunk(dimension(ℓ))

function Base.copy(fℓ::ForwardDiffLogDensity{L,C,T,<:ForwardDiff.GradientConfig}) where {L,C,T}
    @unpack ℓ, chunk, tag, gradient_config = fℓ
    ForwardDiffLogDensity(ℓ, chunk, tag, copy(gradient_config))
end

"""
$(SIGNATURES)

Make a `ForwardDiff.GradientConfig` for type `T`. `tag = nothing` generates the default tag.

Return the function for evaluating log density (with a vector argument) as the second value.
"""
function _make_gradient_config(::Type{T}, ℓ, chunk, tag) where T
    ℓ′ = Base.Fix1(logdensity, ℓ)
    x = zeros(T, dimension(ℓ))
    c = _chunk(chunk)
    gradient_config = if tag ≡ nothing
        ForwardDiff.GradientConfig(ℓ′, x, c)
    else
        ForwardDiff.GradientConfig(ℓ′, x, c, tag)
    end
    gradient_config, ℓ′
end

"""
    ADgradient(:ForwardDiff, ℓ; x, chunk, gradientconfig)
    ADgradient(Val(:ForwardDiff), ℓ; x, chunk, gradientconfig)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keyword arguments:

- `chunk` can be used to set the chunk size, an integer or a `ForwardDiff.Chunk`

- `tag` (default: `nothing`) can be used to set a tag for `ForwardDiff`

- `gradient_config_type` (default: `Union{}`) can be used to preallocate a
  `ForwardDiff.GradientConfig` for the given type. With the default, one is created for each
  evaluation.

   Note **pre-allocating a `ForwardDiff.GradienConfig` is not thread-safe**. You can
   [`copy`](@ref) the results for concurrent evaluation:
   ```julia
   ∇ℓ1 = ADgradient(:ForwardDiff, ℓ; gradient_config = my_gradient_config)
   ∇ℓ2 = copy(∇ℓ1) # you can now use both, in different threads
   ```
"""
function ADgradient(::Val{:ForwardDiff}, ℓ;
                    chunk::Union{Integer,ForwardDiff.Chunk} = _default_chunk(ℓ),
                    tag::Union{Nothing,ForwardDiff.Tag} = nothing,
                    gradient_config_type::Type{T} = Union{}) where {T}
    gradient_config = if gradient_config_type ≡ Union{}
        nothing
    else
        first(_make_gradient_config(T, ℓ, chunk, tag))
    end
    ForwardDiffLogDensity(ℓ, chunk, tag, gradient_config)
end

function logdensity_and_gradient(fℓ::ForwardDiffLogDensity, x::AbstractVector)
    @unpack ℓ, chunk, tag, gradient_config = fℓ
    buffer = _diffresults_buffer(x)
    if gradient_config ≡ nothing
        gradient_config, ℓ′ = _make_gradient_config(eltype(buffer), ℓ, chunk, tag)
    else
        ℓ′ = Base.Fix1(logdensity, ℓ)
    end
    result = ForwardDiff.gradient!(buffer, ℓ′, x, gradient_config)
    _diffresults_extract(result)
end

end # module
