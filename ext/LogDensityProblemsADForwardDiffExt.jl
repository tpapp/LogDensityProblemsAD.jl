module LogDensityProblemsADForwardDiffExt

using ADTypes: AutoForwardDiff
using LogDensityProblemsAD: LogDensityProblemsAD, ADgradient, dimension
using ForwardDiff: Chunk

_get_chunksize(::Chunk{C}) where {C} = C
_get_chunksize(chunk::Integer) = chunk

_default_chunk(ℓ) = _get_chunksize(dimension(ℓ))

function LogDensityProblemsAD.ADgradient(
    ::Val{:ForwardDiff},
    ℓ;
    chunk::Union{Integer,Chunk} = _default_chunk(ℓ),
    tag = nothing,
    x::Union{Nothing,AbstractVector} = nothing,
)
    chunksize = _get_chunksize(chunk)
    backend = AutoForwardDiff{chunksize,typeof(tag)}(tag)
    if isnothing(x)
        return ADgradient(backend, ℓ)
    else
        return ADgradient(backend, ℓ, x)
    end
end

end # module
