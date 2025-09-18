@inline function mhdrpass(
    t_var::SecondAdjointNodeVar,
    t_par::SecondAdjointParameterNode,
    comp_mixed,
    vals::AbstractVector,
    o2mixed,
    cnt,
    adj,
)
    @inbounds vals[o2mixed+comp_mixed(cnt += 1)] += adj
    cnt
end

@inline function mhdrpass(
    t_par::SecondAdjointParameterNode,
    t_var::SecondAdjointNodeVar,
    comp_mixed,
    vals::AbstractVector,
    o2mixed,
    cnt,
    adj,
)
    @inbounds vals[o2mixed+comp_mixed(cnt += 1)] += adj
    cnt
end

@inline function mhdrpass(
    t_var::SecondAdjointNodeVar,
    t_par::SecondAdjointParameterNode,
    comp_mixed::Nothing,
    pairs_seq::C,
    o2mixed,
    cnt,
    adj,
) where C
    cnt += 1
    push!(pairs_seq, (t_var.i, t_par.i))
    cnt
end

@inline function mhdrpass(
    t_par::SecondAdjointParameterNode,
    t_var::SecondAdjointNodeVar,
    comp_mixed::Nothing,
    pairs_seq::C,
    o2mixed,
    cnt,
    adj,
) where C
    cnt += 1
    push!(pairs_seq, (t_var.i, t_par.i))
    cnt
end

# vp
@inline function mhdrpass(
    t_var::SecondAdjointNodeVar,
    t_par::SecondAdjointParameterNode,
    comp_mixed,
    rows::V,
    cols::V,
    o2mixed,
    cnt,
    adj,
) where {I<:Integer,V<:AbstractVector{I}}
    ind = o2mixed + comp_mixed(cnt += 1)
    @inbounds rows[ind] = t_var.i
    @inbounds cols[ind] = t_par.i
    cnt
end
@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed,
    y1,
    y2::T,
    o2mixed,
    cnt,
    adj,
) where T
    cnt += 1
    push!(y2, ((t1.i, t2.i), o2mixed + comp_mixed(cnt)))
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed::Nothing,
    y1,
    y2::T,
    o2mixed,
    cnt,
    adj,
) where T
    cnt += 1
    push!(y2, (t1.i, t2.i))
    cnt
end
@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed,
    y1::AbstractVector,
    y2::Nothing,
    o2mixed,
    cnt,
    adj,
)
    cnt += 1
    @inbounds y1[o2mixed + comp_mixed(cnt)] += adj
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed,
    y1::Tuple,
    y2::Nothing,
    o2mixed,
    cnt,
    adj,
)
    cnt += 1
    Hmtv, v = y1
    @inbounds Hmtv[o2mixed + comp_mixed(cnt)] += adj * v[t1.i]
    cnt
end
# pv
@inline function mhdrpass(
    t_par::SecondAdjointParameterNode,
    t_var::SecondAdjointNodeVar,
    comp_mixed,
    rows::V,
    cols::V,
    o2mixed,
    cnt,
    adj,
) where {I<:Integer,V<:AbstractVector{I}}
    ind = o2mixed + comp_mixed(cnt += 1)
    @inbounds rows[ind] = t_var.i
    @inbounds cols[ind] = t_par.i
    cnt
end
@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed,
    y1,
    y2::T,
    o2mixed,
    cnt,
    adj,
) where T
    cnt += 1
    push!(y2, ((t2.i, t1.i), o2mixed + comp_mixed(cnt)))
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed::Nothing,
    y1,
    y2::T,
    o2mixed,
    cnt,
    adj,
) where T
    cnt += 1
    push!(y2, (t2.i, t1.i))
    cnt
end
@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed,
    y1::AbstractVector,
    y2::Nothing,
    o2mixed,
    cnt,
    adj,
)
    cnt += 1
    @inbounds y1[o2mixed + comp_mixed(cnt)] += adj
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed,
    y1::Tuple,
    y2::Nothing,
    o2mixed,
    cnt,
    adj,
)
    cnt += 1
    Hmtv, v = y1
    @inbounds Hmtv[o2mixed + comp_mixed(cnt)] += adj * v[t2.i]
    cnt
end

#?a₁
@inline function mhdrpass(
    t1,
    t2::SecondAdjointNode1,
    comp_mixed,
    y1,
    y2,
    o2mixed,
    cnt,
    adj,
)
    cnt = mhdrpass(t1, t2.inner, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y)
    cnt
end

#a₁?
@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2,
    comp_mixed,
    y1,
    y2,
    o2mixed,
    cnt,
    adj,
)
    cnt = mhdrpass(t1.inner, t2, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y)
    cnt
end

#a₁a₁
@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode1,
    comp_mixed,
    y1,
    y2,
    o2mixed,
    cnt,
    adj,
)
    cnt = mhdrpass(t1.inner, t2.inner, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y * t2.y)
    cnt
end

#?a₂
@inline function mhdrpass(
    t1,
    t2::SecondAdjointNode2,
    comp_mixed,
    y1,
    y2,
    o2mixed,
    cnt,
    adj,
)
    cnt = mhdrpass(t1, t2.inner1, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y1)
    cnt = mhdrpass(t1, t2.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y2)
    cnt
end

#a₂?
@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2,
    comp_mixed,
    y1,
    y2,
    o2mixed,
    cnt,
    adj,
)
    cnt = mhdrpass(t1.inner1, t2, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y1)
    cnt = mhdrpass(t1.inner2, t2, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y2)
    cnt
end

#a₂a₁
@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode1,
    comp_mixed,
    y1,
    y2,
    o2mixed,
    cnt,
    adj,
)
    cnt = mhdrpass(t1, t2.inner, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y)
    cnt
end

#a₂a₂
@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode2,
    comp_mixed,
    y1,
    y2,
    o2mixed,
    cnt,
    adj,
)
    cnt = mhdrpass(t1, t2.inner1, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y1)
    cnt = mhdrpass(t1, t2.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y2)
    cnt
end

# vv, pp, null, scalar -> noop
@inline mhdrpass(t1::SecondAdjointNodeVar, t2::SecondAdjointNodeVar, comp_mixed, y1, y2, o2mixed, cnt, adj) = cnt
@inline mhdrpass(t1::SecondAdjointParameterNode, t2::SecondAdjointParameterNode, comp_mixed, y1, y2, o2mixed, cnt, adj) = cnt
@inline mhdrpass(t1::Union{SecondAdjointNull,Real}, t2, comp_mixed, y1, y2, o2mixed, cnt, adj) = cnt
@inline mhdrpass(t1, t2::Union{SecondAdjointNull,Real}, comp_mixed, y1, y2, o2mixed, cnt, adj) = cnt

# mh{t}prod
@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed,
    y1::Tuple{V1,V2},
    y2::Nothing,
    o2mixed,
    cnt,
    adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt += 1
    result, input = y1
    @inbounds result[t1.i] += adj * input[t2.i]
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed,
    y1::Tuple{V1,V2},
    y2::Nothing,
    o2mixed,
    cnt,
    adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt += 1
    result, input = y1
    @inbounds result[t2.i] += adj * input[t1.i]
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed,
    y1::Nothing,
    y2::Tuple{V1,V2},
    o2mixed,
    cnt,
    adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt += 1
    result, input = y2
    @inbounds result[t2.i] += adj * input[t1.i]
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed,
    y1::Nothing,
    y2::Tuple{V1,V2},
    o2mixed,
    cnt,
    adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt += 1
    result, input = y2
    @inbounds result[t1.i] += adj * input[t2.i]
    cnt
end



@inline function mhrpass0(t::SecondAdjointNode1, comp_mixed, y1, y2, o2mixed, cnt, adj)
    mhrpass0(t.inner, comp_mixed, y1, y2, o2mixed, cnt, adj * t.y)
end
@inline function mhrpass0(t::SecondAdjointNode1, comp_mixed::Nothing, pairs_seq::Vector{Any}, o2mixed::Int, cnt::Int, adj)
    mhrpass0(t.inner, comp_mixed, pairs_seq, o2mixed, cnt, adj * t.y)
end
@inline function mhrpass0(t::SecondAdjointNode1, comp_mixed, y1, o2mixed, cnt, adj)
    mhrpass0(t.inner, comp_mixed, y1, nothing, o2mixed, cnt, adj * t.y)
end


@inline function mhrpass0(t::SecondAdjointNode2, comp_mixed, y1, y2, o2mixed, cnt, adj)
    cnt = mhrpass0(t.inner1, comp_mixed, y1, y2, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj * t.y2)
    cnt = mhdrpass(t.inner1, t.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj * t.h12)
    cnt
end
@inline function mhrpass0(t::SecondAdjointNode2, comp_mixed::Nothing, pairs_seq::Vector{Any}, o2mixed::Int, cnt::Int, adj)
    cnt = mhrpass0(t.inner1, comp_mixed, pairs_seq, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp_mixed, pairs_seq, o2mixed, cnt, adj * t.y2)
    cnt = mhdrpass(t.inner1, t.inner2, comp_mixed, nothing, pairs_seq, o2mixed, cnt, adj * t.h12)
    cnt
end
@inline function mhrpass0(t::SecondAdjointNode2, comp_mixed, y1, o2mixed, cnt, adj)
    cnt = mhrpass0(t.inner1, comp_mixed, y1, nothing, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp_mixed, y1, nothing, o2mixed, cnt, adj * t.y2)
    cnt = mhdrpass(t.inner1, t.inner2, comp_mixed, y1, nothing, o2mixed, cnt, adj * t.h12)
    cnt
end


@inline function mhrpass0(
    t::SecondAdjointNode2{F}, comp_mixed, y1, y2, o2mixed, cnt, adj
) where {F<:Union{typeof(+),typeof(-)}}
    cnt = mhrpass0(t.inner1, comp_mixed, y1, y2, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj * t.y2)
    cnt
end
@inline function mhrpass0(
    t::SecondAdjointNode2{F}, comp_mixed::Nothing, pairs_seq::Vector{Any}, o2mixed::Int, cnt::Int, adj
) where {F<:Union{typeof(+),typeof(-)}}
    cnt = mhrpass0(t.inner1, comp_mixed, pairs_seq, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp_mixed, pairs_seq, o2mixed, cnt, adj * t.y2)
    cnt
end
@inline function mhrpass0(
    t::SecondAdjointNode2{F}, comp_mixed, y1, o2mixed, cnt, adj
) where {F<:Union{typeof(+),typeof(-)}}
    cnt = mhrpass0(t.inner1, comp_mixed, y1, nothing, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp_mixed, y1, nothing, o2mixed, cnt, adj * t.y2)
    cnt
end

@inline function mhrpass0(t, comp_mixed, y2::Tuple{V,V}, o2mixed, cnt, adj) where {I<:Integer,V<:AbstractVector{I}}
    rows, cols = y2
    mhrpass0(t, comp_mixed, rows, cols, o2mixed, cnt, adj)
end


@inline mhrpass0(t::Union{SecondAdjointNodeVar,SecondAdjointParameterNode,SecondAdjointNull,Real}, comp_mixed, y1, y2, o2mixed, cnt, adj) = cnt
@inline mhrpass0(t::Union{SecondAdjointNodeVar,SecondAdjointParameterNode,SecondAdjointNull,Real}, comp_mixed, y1, o2mixed, cnt, adj) = cnt

@inline function mhrpass0(
    t::T,
    comp,
    y1::Tuple{V1,V2},
    o2mixed,
    cnt,
    adj,
) where {T<:Union{SecondAdjointNull,Real,SecondAdjointParameterNode,SecondAdjointNodeVar},V1<:AbstractVector,V2<:AbstractVector}
    cnt
end

@inline function mhrpass0(
    t::T,
    comp,
    y1::Tuple{V1,V2},
    o2mixed,
    cnt,
    adj,
) where {T<:SecondAdjointNode1,V1<:AbstractVector,V2<:AbstractVector}
    mhrpass0(t.inner, comp, y1, o2mixed, cnt, adj * t.y)
end

@inline function mhrpass0(
    t::T,
    comp,
    y1::Tuple{V1,V2},
    o2mixed,
    cnt,
    adj,
) where {T<:SecondAdjointNode2,V1<:AbstractVector,V2<:AbstractVector}
    cnt = mhrpass0(t.inner1, comp, y1, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp, y1, o2mixed, cnt, adj * t.y2)
    cnt = mhdrpass(t.inner1, t.inner2, comp, y1, nothing, o2mixed, cnt, adj * t.h12)
    cnt
end

@inline function mhrpass0(
    t::T,
    comp,
    y1::Tuple{V1,V2},
    o2mixed,
    cnt,
    adj,
) where {F<:Union{typeof(+),typeof(-)},T<:SecondAdjointNode2{F},V1<:AbstractVector,V2<:AbstractVector}
    cnt = mhrpass0(t.inner1, comp, y1, o2mixed, cnt, adj * t.y1)
    cnt = mhrpass0(t.inner2, comp, y1, o2mixed, cnt, adj * t.y2)
    cnt
end


"""
    smhessian!(y1, y2, f, x, θ, adj)

Performs sparse mixed Hessian evaluation (∂²f/∂x∂θ)

# Arguments:
- `y1`: result vector #1
- `y2`: result vector #2 (only used when evaluating sparsity) 
- `f`: the function to be differentiated in `SIMDFunction` format
- `x`: variable vector
- `θ`: parameter vector
- `adj`: initial adjoint (scalar or vector)
"""
function smhessian!(y1, y2, f, x, θ, adj)
    @simd for k in eachindex(f.itr)
        @inbounds smhessian!(
            y1,
            y2,
            f.f,
            f.itr[k],
            x,
            θ,
            f.f.mcomp2,
            moffset2(f, k),
            isa(adj, AbstractVector) ? adj[offset0(f, k)] : adj,
        )
    end
end

function smhessian!(y1::Nothing, y2::Tuple{VI,VI}, f, p, x, θ, comp, o2mixed, adj) where {I<:Integer,VI<:AbstractVector{I}}
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    rows, cols = y2
    mhrpass0(graph, comp, rows, cols, o2mixed, 0, adj)
end
function smhessian!(y1::AbstractVector, y2::Nothing, f, p, x, θ, comp, o2mixed, adj)
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    mhrpass0(graph, comp, y1, o2mixed, 0, adj)
end
function smhessian!(y1, y2, f, p, x, θ, comp, o2mixed, adj)
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    mhrpass0(graph, comp, y1, y2, o2mixed, 0, adj)
end

# Structure computation overloads that handle nothing values for x and θ
function smhessian!(y1::Nothing, y2::Tuple{VI,VI}, f, p, x::Nothing, θ::Nothing, comp, o2mixed, adj) where {I<:Integer,VI<:AbstractVector{I}}
    graph = f(p, SecondAdjointNodeSource(nothing), SecondAdjointParameterSource(nothing))
    rows, cols = y2
    mhrpass0(graph, comp, rows, cols, o2mixed, 0, adj)
end
function smhessian!(y1::AbstractVector, y2::Nothing, f, p, x::Nothing, θ::Nothing, comp, o2mixed, adj)
    graph = f(p, SecondAdjointNodeSource(nothing), SecondAdjointParameterSource(nothing))
    mhrpass0(graph, comp, y1, o2mixed, 0, adj)
end
function smhessian!(y1, y2, f, p, x::Nothing, θ::Nothing, comp, o2mixed, adj)
    graph = f(p, SecondAdjointNodeSource(nothing), SecondAdjointParameterSource(nothing))
    mhrpass0(graph, comp, y1, y2, o2mixed, 0, adj)
end
