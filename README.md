# LogDensityProblemsAD.jl

![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
[![build](https://github.com/tpapp/LogDensityProblemsAD.jl/workflows/CI/badge.svg)](https://github.com/tpapp/LogDensityProblemsAD.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/LogDensityProblemsAD.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/LogDensityProblemsAD.jl?branch=master)

<!-- Documentation -- uncomment or delete as needed -->
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/LogDensityProblemsAD.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/LogDensityProblemsAD.jl/dev)
-->

Automatic differentiation backends for [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl).

The only exposed function is `ADgradient`. Example:

```julia
using LogDensityProblemsAD, ForwardDiff
∇ℓ = ADgradient(:ForwardDiff, ℓ) # assumes ℓ implements the LogDensityProblems interface
```

Currently, the following backends are supported:

| backend                                                       | notes                                         |
|---------------------------------------------------------------|-----------------------------------------------|
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) | robust, but not ideal for ℝⁿ→ℝ functions      |
| [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) |                                               |
| [Zygote.jl](https://fluxml.ai/Zygote.jl/latest/)              |                                               |
| [Enzyme.jl](https://enzyme.mit.edu/julia/)                    | experimental                                  |
| [Tracker.jl](https://github.com/FluxML/Tracker.jl)            | not heavily maintained, you may prefer Zygote |
