"""
Automatic differentiation backends for LogDensityProblems.
"""
module LogDensityProblemsAD

using ADTypes
import DifferentiationInterface as DI
using DocStringExtensions: SIGNATURES
using LogDensityProblems:
    LogDensityProblems,
    LogDensityOrder,
    logdensity,
    logdensity_and_gradient,
    capabilities,
    dimension
using PackageExtensionCompat: @require_extensions

export ADgradient

## Internal type

struct ADgradientDI{B<:AbstractADType,L,E<:Union{DI.GradientExtras,Nothing}}
    backend::B
    ℓ::L
    extras::E
end

LogDensityProblems.logdensity(∇ℓ::ADgradientDI, x::AbstractVector) = logdensity(∇ℓ.ℓ, x)
LogDensityProblems.capabilities(::Type{<:ADgradientDI}) = LogDensityOrder{1}()
LogDensityProblems.dimension(∇ℓ::ADgradientDI) = dimension(∇ℓ.ℓ)
Base.parent(∇ℓ::ADgradientDI) = ∇ℓ.ℓ
Base.copy(∇ℓ::ADgradientDI) = deepcopy(∇ℓ)


function LogDensityProblems.logdensity_and_gradient(
    ∇ℓ::ADgradientDI{<:Any,<:Any,Nothing},
    x::AbstractVector,
)
    return DI.value_and_gradient(Base.Fix1(logdensity, ∇ℓ.ℓ), ∇ℓ.backend, x)
end

function LogDensityProblems.logdensity_and_gradient(
    ∇ℓ::ADgradientDI{<:Any,<:Any,<:DI.GradientExtras},
    x::AbstractVector,
)
    return DI.value_and_gradient(Base.Fix1(logdensity, ∇ℓ.ℓ), ∇ℓ.backend, x, ∇ℓ.extras)
end

## Constructor from ADTypes

function ADgradient(backend::AbstractADType, ℓ)
    return ADgradientDI(backend, ℓ, nothing)
end

function ADgradient(backend::AbstractADType, ℓ, x::AbstractVector)
    extras = DI.prepare_gradient(Base.Fix1(logdensity, ℓ), backend, x)
    return ADgradientDI(backend, ℓ, extras)
end

## Constructor from symbols

function ADgradient(kind::Symbol, ℓ; kwargs...)
    return ADgradient(Val{kind}(), ℓ; kwargs...)
end

function ADgradient(v::Val{kind}, ℓ; kwargs...) where {kind}
    @info "Don't know how to AD with $(kind), consider `import $(kind)` if there is such a package."
    throw(MethodError(ADgradient, (v, ℓ)))
end

function ADgradient(
    ::Val{:ReverseDiff},
    ℓ;
    compile::Val{comp} = Val(false),
    x::Union{AbstractVector,Nothing} = nothing,
) where {comp}
    backend = AutoReverseDiff(; compile = comp)
    if isnothing(x)
        return ADgradient(backend, ℓ)
    else
        return ADgradient(backend, ℓ, x)
    end
end

function ADgradient(::Val{:Tracker}, ℓ)
    return ADgradient(AutoTracker(), ℓ)
end

function ADgradient(::Val{:Zygote}, ℓ)
    return ADgradient(AutoZygote(), ℓ)
end

## Initialization

function __init__()
    @require_extensions
end

end # module
