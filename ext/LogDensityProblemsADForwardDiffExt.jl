"""
Gradient AD implementation using ForwardDiff.
"""
module LogDensityProblemsADForwardDiffExt

using LogDensityProblemsAD: ADGradientWrapper, SIGNATURES, dimension, logdensity

using ADTypes
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

function ADgradient(ad::AutoForwardDiff{C}, ℓ;
                    x::Union{Nothing,AbstractVector} = nothing) where C
    (; tag) = ad
    _chunk = ForwardDiff.Chunk(something(C, dimension(ℓ))) # will cap chunk size to default
    gradient_config = if x ≡ nothing
        nothing
    else
        _make_gradient_config(Base.Fix1(logdensity, ℓ), x, _chunk, tag)
    end
    ForwardDiffLogDensity(ℓ, _chunk, tag, gradient_config)
end

@deprecate(ADgradient(::Val{:ForwardDiff}, ℓ; x = nothing, chunk = nothing, tag = nothing),
           ADgradient(AutoForwardDiff(; tag, chunksize = chunk), ℓ; x))

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
