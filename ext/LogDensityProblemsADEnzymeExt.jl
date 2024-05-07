module LogDensityProblemsADEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Reverse
if isdefined(Base, :get_extension)
    using LogDensityProblemsAD: LogDensityProblemsAD, ADgradient, logdensity
else
    using ..LogDensityProblemsAD: LogDensityProblemsAD, ADgradient, logdensity
end

function LogDensityProblemsAD.ADgradient(
    ::Val{:Enzyme},
    ℓ;
    mode=Reverse,
    shadow=nothing,
)
    if !isnothing(shadow)
        @warn "keyword argument `shadow` is now ignored"
    end
    backend = AutoEnzyme(; mode)
    return ADgradient(backend, ℓ)
end

end # module
