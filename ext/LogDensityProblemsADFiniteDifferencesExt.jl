module LogDensityProblemsADFiniteDifferencesExt

using ADTypes: AutoFiniteDifferences
using LogDensityProblemsAD: LogDensityProblemsAD, ADgradient
import FiniteDifferences: central_fdm

function LogDensityProblemsAD.ADgradient(::Val{:FiniteDifferences}, ℓ)
    fdm = central_fdm(5, 1)
    backend = AutoFiniteDifferences(; fdm)
    ADgradient(backend, ℓ)
end

end # module
