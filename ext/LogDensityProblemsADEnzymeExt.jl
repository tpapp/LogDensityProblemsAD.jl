module LogDensityProblemsADEnzymeExt

using ADTypes: AutoEnzyme
using LogDensityProblemsAD: ADgradient, logdensity
using Enzyme: Enzyme

function ADgradient(::Val{:Enzyme}, ℓ; mode = Enzyme.Reverse, shadow = nothing)
    @info "keyword argument `shadow` is now ignored"
    backend = AutoEnzyme(; mode)
    return ADgradient(backend, ℓ)
end

end # module
