module LogDensityProblemsADEnzymeExt

using ADTypes: AutoEnzyme
using LogDensityProblemsAD: LogDensityProblemsAD, ADgradient, logdensity
using Enzyme: Enzyme

function LogDensityProblemsAD.ADgradient(
    ::Val{:Enzyme},
    ℓ;
    mode = Enzyme.Reverse,
    shadow = nothing,
)
    if !isnothing(shadow)
        @warn "keyword argument `shadow` is now ignored"
    end
    backend = AutoEnzyme(; mode)
    return ADgradient(backend, ℓ)
end

end # module
