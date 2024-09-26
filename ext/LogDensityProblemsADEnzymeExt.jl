"""
Gradient AD implementation using Enzyme.
"""
module LogDensityProblemsADEnzymeExt

if isdefined(Base, :get_extension)
    using LogDensityProblemsAD: ADGradientWrapper, logdensity

    import LogDensityProblemsAD: ADgradient, logdensity_and_gradient
    import Enzyme
else
    using ..LogDensityProblemsAD: ADGradientWrapper, logdensity

    import ..LogDensityProblemsAD: ADgradient, logdensity_and_gradient
    import ..Enzyme
end

struct EnzymeGradientLogDensity{L,M<:Union{Enzyme.ForwardMode,Enzyme.ReverseMode},S} <: ADGradientWrapper
    ℓ::L
    mode::M
    shadow::S # only used in forward mode
end

"""
    ADgradient(:Enzyme, ℓ; kwargs...)
    ADgradient(Val(:Enzyme), ℓ; kwargs...)

Gradient using algorithmic/automatic differentiation via Enzyme.

# Keyword arguments

- `mode::Enzyme.Mode`: Differentiation mode (default: `Enzyme.Reverse`).
  Currently only `Enzyme.Reverse` and `Enzyme.Forward` are supported.

- `shadow`: Collection of one-hot vectors for each entry of the inputs `x` to the log density
  `ℓ`, or `nothing` (default: `nothing`). This keyword argument is only used in forward
  mode. By default, it will be recomputed in every call of `logdensity_and_gradient(ℓ, x)`.
  For performance reasons it is recommended to compute it only once when calling `ADgradient`.
  The one-hot vectors can be constructed, e.g., with `Enzyme.onehot(x)`.
"""
function ADgradient(::Val{:Enzyme}, ℓ; mode::Enzyme.Mode = Enzyme.Reverse, shadow = nothing)
    mode isa Union{Enzyme.ForwardMode,Enzyme.ReverseMode} ||
        throw(ArgumentError("currently automatic differentiation via Enzyme only supports " *
                            "`Enzyme.Forward` and `Enzyme.Reverse` modes"))
    if mode isa Enzyme.ReverseMode && shadow !== nothing
        @info "keyword argument `shadow` is ignored in reverse mode"
        shadow = nothing

    end
    return EnzymeGradientLogDensity(ℓ, _set_needs_primal(mode), shadow)
end

function _set_needs_primal(ad::Enzyme.ReverseMode{PR, RT, ABI, Holo, Err}) where {PR, RT, ABI, Holo, Err}
    PR && return ad
    return Enzyme.ReverseMode{true, RT, ABI, Holo, Err}()
end

function _set_needs_primal(ad::Enzyme.ForwardMode{PR, ABI, Err, RT}) where {PR, ABI, Err, RT}
    PR && return ad
    return Enzyme.ForwardMode{true, ABI, Err, RT}()
end

function _set_needs_primal(ad::Enzyme.Mode) 
    throw(ArgumentError("Enzyme mode ($ad) is not supported."*
                        "Currently only `ReverseMode` and `ForwardMode` are supported."))
end

function Base.show(io::IO, ∇ℓ::EnzymeGradientLogDensity)
    print(io, "Enzyme AD wrapper for ", ∇ℓ.ℓ, " with ",
          ∇ℓ.mode isa Enzyme.ForwardMode ? "forward" : "reverse", " mode")
end

function logdensity_and_gradient(∇ℓ::EnzymeGradientLogDensity{<:Any,<:Enzyme.ForwardMode},
                                 x::AbstractVector)
    (; ℓ, mode, shadow) = ∇ℓ
    _shadow = shadow === nothing ? Enzyme.onehot(x) : shadow
    ∂ℓ_∂x, y = Enzyme.autodiff(mode, logdensity, Enzyme.BatchDuplicated,
                               Enzyme.Const(ℓ),
                               Enzyme.BatchDuplicated(x, _shadow))
    return y, collect(∂ℓ_∂x)
end

function logdensity_and_gradient(∇ℓ::EnzymeGradientLogDensity{<:Any,<:Enzyme.ReverseMode{WP, RT}},
                                 x::AbstractVector) where {WP, RT}
    (; ℓ) = ∇ℓ
    ∂ℓ_∂x = zero(x)
    mode = ∇ℓ.mode
    _, y = Enzyme.autodiff(mode, logdensity, Enzyme.Active,
                           Enzyme.Const(ℓ), Enzyme.Duplicated(x, ∂ℓ_∂x))
    y, ∂ℓ_∂x
end

end # module
