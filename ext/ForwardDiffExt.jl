"""
Gradient AD implementation using ForwardDiff.
"""
module ForwardDiffExt

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

struct ForwardDiffLogDensity{L, C <: ForwardDiff.Chunk,
                             G <: Union{Nothing,ForwardDiff.GradientConfig}} <: ADGradientWrapper
    "supports zero-order evaluation `logdensity(ℓ, x)`"
    ℓ::L
    "chunk size for ForwardDiff"
    chunk::C
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

function _make_gradient_config(::Type{T}, ℓ, chunk) where T
    ForwardDiff.GradientConfig(Base.Fix1(logdensity, ℓ), zeros(T, dimension(ℓ)),
                               _chunk(chunk))
end

function Base.copy(fℓ::ForwardDiffLogDensity{L,C,<:ForwardDiff.GradientConfig{T}}) where {L,C,T}
    @unpack ℓ, chunk = fℓ
    gradient_config =_make_gradient_config(T, ℓ, chunk)
    ForwardDiffLogDensity(ℓ, chunk, gradient_config)
end

"""
    ADgradient(:ForwardDiff, ℓ; x, chunk, gradientconfig)
    ADgradient(Val(:ForwardDiff), ℓ; x, chunk, gradientconfig)

Wrap a log density that supports evaluation of `Value` to handle `ValueGradient`, using
`ForwardDiff`.

Keyword arguments:

- `chunk` can be used to set the chunk size, an integer or a `ForwardDiff.Chunk`

- `gradient_config_type` can be `nothing` (the default) or a type (eg `Float64`).

   The latter preallocates and reuses a `ForwardDiff.GradientConfig` for that type. Note
   that **this option is not thread-safe**. You can [`copy`](@ref) the results for
   concurrent evaluation:
   ```julia
   ∇ℓ1 = ADgradient(:ForwardDiff, ℓ; gradient_config_type = Float64)
   ∇ℓ2 = copy(∇ℓ1) # you can now use both, in different threads
   ```
"""
function ADgradient(::Val{:ForwardDiff}, ℓ;
                    x::Union{Nothing,AbstractVector} = nothing,
                    chunk::Union{Integer,ForwardDiff.Chunk} = _default_chunk(ℓ),
                    gradient_config_type::Union{Nothing,Type{T}} = nothing) where {T<:Real}
    gradient_config = if gradient_config_type ≡ nothing
        nothing
    else
        S = gradient_config_type
        (isconcretetype(T) && (S <: Real)) ||
            throw(ArgumentError("gradient_config_type needs to be a concrete subtype of Real."))
        _make_gradient_config(S, ℓ, chunk)
    end
    ForwardDiffLogDensity(ℓ, chunk, gradient_config)
end

function logdensity_and_gradient(fℓ::ForwardDiffLogDensity, x::AbstractVector)
    @unpack ℓ, chunk, gradient_config = fℓ
    buffer = _diffresults_buffer(x)
    if gradient_config ≡ nothing
        gradient_config = _make_gradient_config(eltype(x), ℓ, chunk)
    end
    result = ForwardDiff.gradient!(buffer, Base.Fix1(logdensity, ℓ), x, gradient_config)
    _diffresults_extract(result)
end

end # module
