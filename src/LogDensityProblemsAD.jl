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

export ADgradient

struct ADgradient{B<:AbstractADType,L,E<:Union{DI.GradientExtras,Nothing}}
    backend::B
    ℓ::L
    extras::E
end

LogDensityProblems.logdensity(g::ADgradient, x::AbstractVector) = logdensity(g.ℓ, x)
LogDensityProblems.capabilities(::Type{<:ADgradient}) = LogDensityOrder{1}()
LogDensityProblems.dimension(g::ADgradient) = dimension(g.ℓ)
Base.parent(g::ADgradient) = g.ℓ
Base.copy(g::ADgradient) = deepcopy(g)

function ADgradient(backend::AbstractADType, ℓ)
    return ADgradient(backend, ℓ, nothing)
end

function ADgradient(backend::AbstractADType, ℓ, x::AbstractVector)
    extras = DI.prepare_gradient(Base.Fix1(logdensity, ℓ), backend, x)
    return ADgradient(backend, ℓ, extras)
end

function LogDensityProblems.logdensity_and_gradient(
    ∇ℓ::ADgradient{<:Any,<:Any,Nothing},
    x::AbstractVector,
)
    (; ℓ, backend) = ∇ℓ
    return DI.value_and_gradient(Base.Fix1(logdensity, ℓ), backend, x)
end

function LogDensityProblems.logdensity_and_gradient(
    ∇ℓ::ADgradient{<:Any,<:Any,<:DI.GradientExtras},
    x::AbstractVector,
)
    (; ℓ, backend, extras) = ∇ℓ
    return DI.value_and_gradient(Base.Fix1(logdensity, ℓ), backend, x, extras)
end

## Translation from symbols

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

end # module
