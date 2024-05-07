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

if !isdefined(Base, :get_extension)
    using Requires: @require
end

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
    compile::Val{comp}=Val(false),
    x::Union{AbstractVector,Nothing}=nothing,
) where {comp}
    backend = AutoReverseDiff(; compile=comp)
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

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin
            include("../ext/LogDensityProblemsADEnzymeExt.jl")
        end
        @require FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000" begin
            include("../ext/LogDensityProblemsADFiniteDifferencesExt.jl")
        end
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/LogDensityProblemsADForwardDiffExt.jl")
        end
    end
end

end # module
