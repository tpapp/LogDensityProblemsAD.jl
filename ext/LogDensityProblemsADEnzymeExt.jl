"""
Gradient AD implementation using Enzyme.
"""
module LogDensityProblemsADEnzymeExt

using LogDensityProblemsAD: ADGradientWrapper, logdensity

using ADTypes: AutoEnzyme
using ArgCheck: @argcheck
import Enzyme
using Enzyme: EnzymeCore
import LogDensityProblemsAD: ADgradient, logdensity_and_gradient

struct EnzymeGradientLogDensity{L,M<:Union{Enzyme.ForwardMode,
                                           Enzyme.ReverseMode}} <: ADGradientWrapper
    ℓ::L
    mode::M
end

function ADgradient(ad::AutoEnzyme, ℓ; x::Union{Nothing,AbstractVector}=nothing)
    (; mode) = ad
    @argcheck(mode isa Union{Nothing,Enzyme.ForwardMode,Enzyme.ReverseMode},
              "currently automatic differentiation via Enzyme only supports " *
                  "`Enzyme.Forward` and `Enzyme.Reverse` modes")
    EnzymeGradientLogDensity(ℓ, Enzyme.WithPrimal(something(mode, Enzyme.Reverse)))
end

function Base.show(io::IO, ∇ℓ::EnzymeGradientLogDensity)
    print(io, "Enzyme AD wrapper for ", ∇ℓ.ℓ, " with ",
          ∇ℓ.mode isa Enzyme.ForwardMode ? "forward" : "reverse", " mode")
end

function logdensity_and_gradient(∇ℓ::EnzymeGradientLogDensity{<:Any,<:Enzyme.ForwardMode},
                                 x::AbstractVector{T}) where T
    (; ℓ, mode) = ∇ℓ
    result = Enzyme.gradient(mode, Base.Fix1(logdensity, ℓ), x)
    T(result.val)::T, collect(T, only(result.derivs))::Vector{T}
end

function logdensity_and_gradient(∇ℓ::EnzymeGradientLogDensity{<:Any,<:Enzyme.ReverseMode},
                                 x::AbstractVector)
    (; ℓ, mode) = ∇ℓ
    ∂ℓ_∂x = zero(x)
    _, y = Enzyme.autodiff(mode, logdensity, Enzyme.Active,
                           Enzyme.Const(ℓ), Enzyme.Duplicated(x, ∂ℓ_∂x))
    y, ∂ℓ_∂x
end

@deprecate(ADgradient(::Val{:Enzyme}, P; mode = Enzyme.Reverse, x = nothing,
                      shadow = nothing),
           ADgradient(AutoEnzyme(; mode), P; x))

end # module
