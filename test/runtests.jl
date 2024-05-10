using LogDensityProblemsAD
using Test, Random
import LogDensityProblems: capabilities, dimension, logdensity
using LogDensityProblems: logdensity_and_gradient, LogDensityOrder
import FiniteDifferences, ForwardDiff, Enzyme, Tracker, Zygote, ReverseDiff # backends
import ADTypes # load support for AD types with options
import BenchmarkTools                            # load the heuristic chunks code
using ComponentArrays: ComponentVector           # test with other vector types

struct EnzymeTestMode <: Enzyme.Mode{Enzyme.DefaultABI} end

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
ForwardDiff.checktag(::Type{ForwardDiff.Tag{TestTag,V}}, ::Base.Fix1{typeof(logdensity),typeof(TestLogDensity())}, ::AbstractArray{V}) where {V} = true

@testset "AD via ReverseDiff" begin
    ℓ = TestLogDensity()

    ∇ℓ_default = ADgradient(:ReverseDiff, ℓ)
    ∇ℓ_nocompile = ADgradient(:ReverseDiff, ℓ; compile=Val(false))

    # ADTypes support
    @test ADgradient(ADTypes.AutoReverseDiff(), ℓ) === ∇ℓ_default
    @test ADgradient(ADTypes.AutoReverseDiff(; compile = false), ℓ) === ∇ℓ_nocompile

    ∇ℓ_compile = ADgradient(:ReverseDiff, ℓ; compile=Val(true))
    ∇ℓ_compile_x = ADgradient(:ReverseDiff, ℓ; compile=Val(true), x=randexp(3))

    # ADTypes support
    @test typeof(ADgradient(ADTypes.AutoReverseDiff(; compile = true), ℓ)) === typeof(∇ℓ_compile)

    for ∇ℓ in (∇ℓ_default, ∇ℓ_nocompile, ∇ℓ_compile, ∇ℓ_compile_x)
        @test dimension(∇ℓ) == 3
        @test capabilities(∇ℓ) ≡ LogDensityOrder(1)

        for _ in 1:100
            x = randexp(3)
            @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity(x)
            @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
                (test_logdensity(x), test_gradient(x))

            x = -x
            @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity(x)
            if ∇ℓ.extras === nothing
                # Recompute tape => correct results
                @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
                    (test_logdensity(x), zero(x))
            else
                # Tape not recomputed => incorrect results, uses always the same branch
                @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
                    (test_logdensity1(x), test_gradient(x))
            end
        end
    end
end

@testset "AD via ForwardDiff" begin
    ℓ = TestLogDensity()
    ∇ℓ = ADgradient(:ForwardDiff, ℓ)
    @test dimension(∇ℓ) == 3
    @test capabilities(∇ℓ) ≡ LogDensityOrder(1)


    # ADTypes support
    @test ADgradient(ADTypes.AutoForwardDiff(; chunksize=dimension(ℓ)), ℓ) === ∇ℓ

    for _ in 1:100
        x = randexp(3)
        @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity(x)
        @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
            (test_logdensity(x), test_gradient(x))
    end

    # custom tag
    for T in (Float32, Float64)
        x = randexp(T, 3)
        for tag in (ForwardDiff.Tag(TestTag(), T), TestTag())
            local ∇ℓ = ADgradient(:ForwardDiff, ℓ; tag = tag)
            @test eltype(first(logdensity_and_gradient(∇ℓ, x))) === T
            @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity(x)
            @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
                (test_logdensity(x), test_gradient(x))
        end
    end

    # preallocated gradient config
    x = randexp(Float32, 3)
    ∇ℓ = ADgradient(:ForwardDiff, ℓ; x = x)
    @test eltype(first(logdensity_and_gradient(∇ℓ, x))) === Float32
    @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity(x)
    @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
        (test_logdensity(x), test_gradient(x))
    @test @inferred(copy(∇ℓ)).extras.config ≢ ∇ℓ.extras.config

    # custom tag + preallocated gradient config
    for T in (Float32, Float64)
        x = randexp(T, 3)
        for tag in (ForwardDiff.Tag(TestTag(), T), TestTag())
            ∇ℓ = ADgradient(:ForwardDiff, ℓ; tag = tag, x = x)
            @test eltype(first(logdensity_and_gradient(∇ℓ, x))) === T
            @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity(x)
            @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅
                (test_logdensity(x), test_gradient(x))
            @test @inferred(copy(∇ℓ)).extras.config ≢ ∇ℓ.extras.config
        end
    end

    # chunk size as integers
    @test ADgradient(:ForwardDiff, ℓ; chunk = 3) isa eltype(∇ℓ)

    # ADTypes support
    @test ADgradient(ADTypes.AutoForwardDiff(; chunksize = 3), ℓ) === ADgradient(:ForwardDiff, ℓ; chunk = 3)
    @test ADgradient(ADTypes.AutoForwardDiff(; chunksize = 3, tag = TestTag()), ℓ) === ADgradient(:ForwardDiff, ℓ; chunk = 3, tag = TestTag())
end

@testset "component vectors" begin
    # test with something else than `Vector`
    # cf https://github.com/tpapp/LogDensityProblemsAD.jl/pull/3
    ℓ = TestLogDensity()
    ∇ℓ = ADgradient(:ForwardDiff, ℓ)
    x = randexp(3)
    y = ComponentVector(x = x)
    @test @inferred(logdensity(∇ℓ, y)) ≅ test_logdensity(x)
    @test @inferred(logdensity_and_gradient(∇ℓ, y)) ≅
        (test_logdensity(x), test_gradient(x))
    ∇ℓ2 = ADgradient(:ForwardDiff, ℓ; x = y) # preallocate GradientConfig
    @test @inferred(logdensity(∇ℓ2, y)) ≅ test_logdensity(x)
    @test @inferred(logdensity_and_gradient(∇ℓ2, y)) ≅
        (test_logdensity(x), test_gradient(x))
end

@testset "AD via Tracker" begin
    ℓ = TestLogDensity()
    ∇ℓ = ADgradient(:Tracker, ℓ)
    @test dimension(∇ℓ) == 3
    @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
    for _ in 1:100
        x = randexp(3)
        @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity(x)
        @test @inferred(logdensity_and_gradient(∇ℓ, x)) ≅ (test_logdensity(x), test_gradient(x))
    end

   # ADTypes support
   @test ADgradient(ADTypes.AutoTracker(), ℓ) === ∇ℓ
end

@testset "AD via Zygote" begin
    ℓ = TestLogDensity(test_logdensity1)
    ∇ℓ = ADgradient(:Zygote, ℓ)
    @test dimension(∇ℓ) == 3
    @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
    for _ in 1:100
        x = randexp(3)
        @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity1(x)
        @test logdensity_and_gradient(∇ℓ, x) ≅ (test_logdensity1(x), test_gradient(x))
    end

   # ADTypes support
   @test ADgradient(ADTypes.AutoZygote(), ℓ) === ∇ℓ
end

@testset "AD via Enzyme" begin
    ℓ = TestLogDensity(test_logdensity1)

    ∇ℓ_reverse = ADgradient(:Enzyme, ℓ)
    @test ∇ℓ_reverse === ADgradient(:Enzyme, ℓ; mode=Enzyme.Reverse)

    # ADTypes support
    @test ADgradient(ADTypes.AutoEnzyme(; mode=Enzyme.Reverse), ℓ) === ∇ℓ_reverse

    ∇ℓ_forward = ADgradient(:Enzyme, ℓ; mode=Enzyme.Forward)
    ∇ℓ_forward_shadow = ADgradient(
        :Enzyme,
        ℓ;
        mode=Enzyme.Forward,
        shadow=Enzyme.onehot(Vector{Float64}(undef, dimension(ℓ))),
    )

    for ∇ℓ in (∇ℓ_reverse, ∇ℓ_forward, ∇ℓ_forward_shadow)
        @test dimension(∇ℓ) == 3
        @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
        for _ in 1:100
            x = randexp(3)
            @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity1(x)
            @test logdensity_and_gradient(∇ℓ, x) ≅ (test_logdensity1(x), test_gradient(x))
        end
    end
end

@testset "AD via FiniteDifferences" begin
    ℓ = TestLogDensity(test_logdensity1)
    ∇ℓ = ADgradient(:FiniteDifferences, ℓ)
    @test dimension(∇ℓ) == 3
    @test capabilities(∇ℓ) ≡ LogDensityOrder(1)
    for _ = 1:100
        x = randexp(3)
        @test @inferred(logdensity(∇ℓ, x)) ≅ test_logdensity1(x)
        @test ≅(logdensity_and_gradient(∇ℓ, x), (test_logdensity1(x), test_gradient(x)); atol = 1e-5)
    end
end

@testset "ADgradient missing method" begin
    msg = "Don't know how to AD with Foo, consider `import Foo` if there is such a package."
    @test_logs((:info, msg), @test_throws(MethodError, ADgradient(:Foo, TestLogDensity2())))
end
