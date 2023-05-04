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

## Backends

Below is the list of supported backends, more or less in the order they are recommended for ℝⁿ→ℝ functions. That said, for nontrivial problems you should do your own benchmarking and compare results from various backends in case you suspect an incorrect calculation (eg because MCMC does not converge and you have ruled everything else out).

1. [Enzyme.jl](https://enzyme.mit.edu/julia/)

Fastest option if it works for your problem, ideal if your code does not allocate. Try it first, with reverse mode (the default).

2. [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)

Robust and mature implementation, but not necessarily the fastest. Scales more or less linearly with input dimension, so use with caution for large problems. Ideal for checking correctness.

3. [Zygote.jl](https://fluxml.ai/Zygote.jl/latest/) and [Tracker.jl](https://github.com/FluxML/Tracker.jl)

May be a viable choice if Enzyme is not working for your problem, and calculations are non-mutating and performed on matrices and vectors, not scalars. Benchmark against alternatives above. Of the two, Zygote is more actively maintained.

4. [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)

At this point the alternatives above are more performant, this is kept for backward compatibility.

PRs for other AD frameworks are welcome, even if they are WIP.
