@inline (a::Pair{P,S} where {P,S<:AbstractNode})(i, x, θ) = a.second(i, x, θ)

"""
    Compressor{I}

Data structure for the sparse index

# Fields:
- `inner::I`: stores the sparse index as a tuple form
"""
struct Compressor{I}
    inner::I
end
@inline (i::Compressor{I})(n) where {I} = @inbounds i.inner[n]

struct SIMDFunction{F,C1,C2,PC1}
    f::F
    comp1::C1
    comp2::C2
    pcomp1::PC1
    o0::Int
    o1::Int
    o2::Int
    po1::Int
    o1step::Int
    o2step::Int
    po1step::Int
end

@inline (sf::SIMDFunction{F,C1,C2,PC1})(i, x, θ) where {F,C1,C2,PC1} = sf.f(i, x, θ)
@inline (sf::SIMDFunction{F,C1,C2,PC1})(i, x, θ) where {F <: Real,C1,C2,PC1} = sf.f

"""
    SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0, po1 = 0)

Returns a `SIMDFunction` using the `gen`.

# Arguments:
- `gen`: an iterable function specified in `Base.Generator` format
- `o0`: offset for the function evaluation
- `o1`: offset for the derivative evalution
- `o2`: offset for the second-order derivative evalution
- `po1`: offset for the parameter derivative evalution
"""
function SIMDFunction(gen::Base.Generator, o0 = 0, o1 = 0, o2 = 0, po1 = 0)

    f = gen.f(ParSource())

    _simdfunction(f, o0, o1, o2, po1)
end

function _simdfunction(f::F, o0, o1, o2, po1) where {F<:Real}
    SIMDFunction(
        f,
        ExaModels.Compressor{Tuple{}}(()),
        ExaModels.Compressor{Tuple{}}(()),
        ExaModels.Compressor{Tuple{}}(()),
        o0,
        o1,
        o2,
        po1,
        0,
        0,
        0,
    )
end

function _simdfunction(f, o0, o1, o2, po1)
    d = f(Identity(), AdjointNodeSource(nothing), nothing)
    y1 = []
    ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

    t = f(Identity(), SecondAdjointNodeSource(nothing), nothing)
    y2 = []
    ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, NaN, NaN)

    a1 = unique(y1)
    o1step = length(a1)
    c1 = Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

    a2 = unique(y2)
    o2step = length(a2)
    c2 = Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

    pd = f(Identity(), nothing, AdjointNodeParameterSource(nothing))
    py1 = []
    ExaModels.grpass(pd, nothing, py1, nothing, 0, NaN)

    pa1 = unique(py1)
    po1step = length(pa1)
    pc1 = Compressor(Tuple(findfirst(isequal(i), pa1) for i in py1))

    SIMDFunction(f, c1, c2, pc1, o0, o1, o2, po1, o1step, o2step, po1step)
end
