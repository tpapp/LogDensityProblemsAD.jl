using LogDensityProblemsAD
using Test, Random
import LogDensityProblems: capabilities, dimension, logdensity
using LogDensityProblems: logdensity_and_gradient, LogDensityOrder
import FiniteDifferences, ForwardDiff, Enzyme, Tracker, Zygote, ReverseDiff # backends
using ADTypes # load support for AD types with options
import DifferentiationInterface
using ComponentArrays: ComponentVector           # test with other vector types

####
#### test setup and utilities
####

###
### reproducible randomness
###

Random.seed!(1)

###
### comparisons (for testing)
###

"""
    a ≅ b

Compare log denfields and types, for unit testing.
"""
≅(::Any, ::Any; atol = 0) = false

function ≅(a::Real, b::Real; atol = 0)
    if isnan(a)
        isnan(b)
    elseif isinf(a)
        a == b
    else
        abs(a - b) ≤ atol
    end
end

function ≅(a::Tuple{Real,Any}, b::Tuple{Real,Any}; atol = 0)
    ≅(first(a), first(b); atol = atol) || return false
    !isfinite(first(a)) || isapprox(last(a), last(b); atol = atol, rtol = 0)
end

###
### simple log densities for testing
###

struct TestLogDensity{F}
    ℓ::F
end
logdensity(ℓ::TestLogDensity, x) = ℓ.ℓ(x)
dimension(::TestLogDensity) = 3
test_logdensity1(x) = -2 * abs2(x[1]) - 3 * abs2(x[2]) - 5 * abs2(x[3])
test_logdensity(x::AbstractVector{T}) where {T} = any(x .< 0) ? -T(Inf) : test_logdensity1(x)
test_gradient(x) = x .* [-4, -6, -10]
TestLogDensity() = TestLogDensity(test_logdensity) # default: -Inf for negative input
Base.show(io::IO, ::TestLogDensity) = print(io, "TestLogDensity")

struct TestLogDensity2 end
logdensity(::TestLogDensity2, x) = -sum(abs2, x)
dimension(::TestLogDensity2) = 20

# Tag type for ForwardDiff
struct TestTag end

# Allow tag type in gradient etc. calls of the log density function
ForwardDiff.checktag(::Type{ForwardDiff.Tag{TestTag, V}}, _, ::AbstractArray{V}) where {V} = true

@testset verbose=true "generic backend tests" begin
    BACKENDS = [
        AutoEnzyme() => "Enzyme defaults",
        AutoEnzyme(; mode = Enzyme.Forward) => "Enzyme w/ Forward",
        AutoEnzyme(; mode = Enzyme.Reverse) => "Enzyme w/ Reverse",
        AutoFiniteDifferences(; fdm = FiniteDifferences.central_fdm(5, 1)) =>
            "FiniteDifferences",
        AutoForwardDiff() => "ForwardDiff w/ defaults",
        AutoForwardDiff(; tag = TestTag()) => "ForwardDiff w/ tag",
        AutoForwardDiff(; chunksize = 1) => "ForwardDiff w/ chunk size",
        AutoReverseDiff(; compile = false) => "ReverseDiff w/o compile",
        AutoReverseDiff(; compile = true) => "ReverseDiff compile",
        AutoTracker() => "Tracker",
        AutoZygote() => "Zygote",
    ]
    ℓ = TestLogDensity(test_logdensity1)
    x = zeros(dimension(ℓ))
    for (backend, backend_label) in BACKENDS
        for (∇ℓ, ∇ℓ_label) in [ADgradient(backend, ℓ) => "$(backend_label) no prep",
                               ADgradient(backend, ℓ; x) => "$(backend_label) w/ prep"]
            @testset "$(∇ℓ_label)" begin
                @test dimension(∇ℓ) == dimension(ℓ)
                @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
                for _ in 1:10
                    x = randn(3)
                    @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity1(x)
                    @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
                        (test_logdensity1(x), test_gradient(x)) atol = 1e-5
                end
            end
        end
    end
end

@testset "Enzyme error for unsupported mode" begin
    struct MockEnzymeMode <: supertype(typeof(Enzyme.Reverse)) end # errors as unsupported
    @test_throws ArgumentError ADgradient(AutoEnzyme(; mode = MockEnzymeMode()),
                                          TestLogDensity(test_logdensity1))
end

@testset "component vectors" begin
    # test with something else than `Vector`
    # cf https://github.com/tpapp/LogDensityProblemsAD.jl/pull/3
    ℓ = TestLogDensity()
    ∇ℓ = ADgradient(:ForwardDiff, ℓ)
    x = zeros(3)
    y = ComponentVector(x = x)
    @test @inferred(logdensity(∇ℓ, y)) ≅ test_logdensity(x)
    @test @inferred(logdensity_and_gradient(∇ℓ, y)) ≅
        (test_logdensity(x), test_gradient(x))
    ∇ℓ2 = ADgradient(:ForwardDiff, ℓ; x = y) # preallocate GradientConfig
    @test @inferred(logdensity(∇ℓ2, y)) ≅ test_logdensity(x)
    @test @inferred(logdensity_and_gradient(∇ℓ2, y)) ≅
        (test_logdensity(x), test_gradient(x))
end
