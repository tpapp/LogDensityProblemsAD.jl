###
### Helper functions for working with DiffResults.
### Only included when required by AD wrappers.
### Requires that `DiffResults` and `DocStringExtensions.SIGNATURES` are available.
###

"""
$(SIGNATURES)

Allocate a DiffResults buffer for a gradient, taking the element type of `x` into account
(heuristically).
"""
function _diffresults_buffer(x)
    T = eltype(x)
    S = T <: Real ? float(T) : Float64 # heuristic
    DiffResults.MutableDiffResult(zero(S), (similar(x, S), ))
end

"""
$(SIGNATURES)

Extract a return value for [`logdensity_and_gradient`](@ref) from a DiffResults buffer,
constructed with [`diffresults_buffer`](@ref). Gradient is not copied as caller created the
vector.
"""
function _diffresults_extract(diffresult::DiffResults.DiffResult)
    DiffResults.value(diffresult), DiffResults.gradient(diffresult)
end
