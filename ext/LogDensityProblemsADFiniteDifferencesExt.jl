module LogDensityProblemsADFiniteDifferencesExt

if isdefined(Base, :get_extension)
    using ADTypes: AutoFiniteDifferences
    import FiniteDifferences: central_fdm
    using LogDensityProblemsAD: LogDensityProblemsAD, ADgradient
else
    using ..ADTypes: AutoFiniteDifferences
    import ..FiniteDifferences: central_fdm
    using ..LogDensityProblemsAD: LogDensityProblemsAD, ADgradient
end

function LogDensityProblemsAD.ADgradient(::Val{:FiniteDifferences}, ℓ)
    fdm = central_fdm(5, 1)
    backend = AutoFiniteDifferences(; fdm)
    ADgradient(backend, ℓ)
end

end # module
