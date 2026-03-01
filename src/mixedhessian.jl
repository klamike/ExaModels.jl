@inline function mhdrpass(
    t_var::SecondAdjointNodeVar,
    t_par::SecondAdjointParameterNode,
    comp_mixed, vals::AbstractVector, o2mixed, cnt, adj,
)
    @inbounds vals[o2mixed+comp_mixed(cnt += 1)] += adj
    cnt
end

@inline function mhdrpass(
    t_par::SecondAdjointParameterNode,
    t_var::SecondAdjointNodeVar,
    comp_mixed, vals::AbstractVector, o2mixed, cnt, adj,
)
    mhdrpass(t_var, t_par, comp_mixed, vals, o2mixed, cnt, adj)
end

@inline function mhdrpass(
    t_var::SecondAdjointNodeVar,
    t_par::SecondAdjointParameterNode,
    comp_mixed::Nothing, pairs_seq::AbstractVector, o2mixed, cnt, adj,
)
    cnt += 1
    push!(pairs_seq, (t_var.i, t_par.i))
    cnt
end

@inline function mhdrpass(
    t_par::SecondAdjointParameterNode,
    t_var::SecondAdjointNodeVar,
    comp_mixed::Nothing, pairs_seq::AbstractVector, o2mixed, cnt, adj,
)
    mhdrpass(t_var, t_par, comp_mixed, pairs_seq, o2mixed, cnt, adj)
end

# vp
@inline function mhdrpass(
    t_var::SecondAdjointNodeVar,
    t_par::SecondAdjointParameterNode,
    comp_mixed, rows::V, cols::V, o2mixed, cnt, adj,
) where {I<:Integer,V<:AbstractVector{I}}
    ind = o2mixed + comp_mixed(cnt += 1)
    @inbounds rows[ind] = t_var.i
    @inbounds cols[ind] = t_par.i
    cnt
end
@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed, y1::Nothing, y2::AbstractVector, o2mixed, cnt, adj,
)
    cnt += 1
    push!(y2, ((t1.i, t2.i), o2mixed + comp_mixed(cnt)))
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed::Nothing, y1::Nothing, y2::AbstractVector, o2mixed, cnt, adj,
)
    cnt += 1
    push!(y2, (t1.i, t2.i))
    cnt
end
@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed, y1::V, y2::Nothing, o2mixed, cnt, adj,
) where {V<:AbstractVector{Tuple{Tuple{Int,Int},Int}}}
    ind = o2mixed + comp_mixed(cnt += 1)
    @inbounds y1[ind] = ((t1.i, t2.i), ind)
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed, y1::AbstractVector, y2::Nothing, o2mixed, cnt, adj,
)
    cnt += 1
    @inbounds y1[o2mixed + comp_mixed(cnt)] += adj
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed, y1::Tuple, y2::Nothing, o2mixed, cnt, adj,
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
    comp_mixed, rows::V, cols::V, o2mixed, cnt, adj,
) where {I<:Integer,V<:AbstractVector{I}}
    mhdrpass(t_var, t_par, comp_mixed, rows, cols, o2mixed, cnt, adj)
end
@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed, y1::Nothing, y2::AbstractVector, o2mixed, cnt, adj,
)
    mhdrpass(t2, t1, comp_mixed, y1, y2, o2mixed, cnt, adj)
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed::Nothing, y1::Nothing, y2::AbstractVector, o2mixed, cnt, adj,
)
    mhdrpass(t2, t1, comp_mixed, y1, y2, o2mixed, cnt, adj)
end
@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed, y1::V, y2::Nothing, o2mixed, cnt, adj,
) where {V<:AbstractVector{Tuple{Tuple{Int,Int},Int}}}
    mhdrpass(t2, t1, comp_mixed, y1, y2, o2mixed, cnt, adj)
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed, y1::AbstractVector, y2::Nothing, o2mixed, cnt, adj,
)
    mhdrpass(t2, t1, comp_mixed, y1, y2, o2mixed, cnt, adj)
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed, y1::Tuple, y2::Nothing, o2mixed, cnt, adj,
)
    mhdrpass(t2, t1, comp_mixed, y1, y2, o2mixed, cnt, adj)
end

#?a₁
@inline function mhdrpass(
    t1,
    t2::SecondAdjointNode1,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
)
    cnt = mhdrpass(t1, t2.inner, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y)
    cnt
end

#a₁?
@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
)
    cnt = mhdrpass(t1.inner, t2, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y)
    cnt
end

#a₁a₁
@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode1,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
)
    cnt = mhdrpass(t1.inner, t2.inner, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y * t2.y)
    cnt
end

#?a₂
@inline function mhdrpass(
    t1,
    t2::SecondAdjointNode2,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
)
    cnt = mhdrpass(t1, t2.inner1, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y1)
    cnt = mhdrpass(t1, t2.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y2)
    cnt
end

#a₂?
@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
)
    cnt = mhdrpass(t1.inner1, t2, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y1)
    cnt = mhdrpass(t1.inner2, t2, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y2)
    cnt
end

#a₂a₁
@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode1,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
)
    cnt = mhdrpass(t1, t2.inner, comp_mixed, y1, y2, o2mixed, cnt, adj * t2.y)
    cnt
end

#a₂a₂
@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode2,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
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
@inline mhdrpass(t1::Union{SecondAdjointNull,Real}, t2::Union{SecondAdjointNull,Real}, comp_mixed, y1, y2, o2mixed, cnt, adj) = cnt
@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode2,
    comp_mixed, y1, y2, o2mixed, cnt, adj,
)
    cnt = mhdrpass(t1.inner, t2, comp_mixed, y1, y2, o2mixed, cnt, adj * t1.y)
    cnt
end

# mh{t}prod
@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed, y1::Tuple{V1,V2}, y2::Nothing, o2mixed, cnt, adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt += 1
    result, input = y1
    @inbounds result[t1.i] += adj * input[t2.i]
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed, y1::Tuple{V1,V2}, y2::Nothing, o2mixed, cnt, adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    mhdrpass(t2, t1, comp_mixed, y1, y2, o2mixed, cnt, adj)
end

@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp_mixed, y1::Nothing, y2::Tuple{V1,V2}, o2mixed, cnt, adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt += 1
    result, input = y2
    @inbounds result[t2.i] += adj * input[t1.i]
    cnt
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp_mixed, y1::Nothing, y2::Tuple{V1,V2}, o2mixed, cnt, adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    mhdrpass(t2, t1, comp_mixed, y1, y2, o2mixed, cnt, adj)
end



@inline mhrpass0(t, comp_mixed, y1, y2, o2mixed, cnt, adj) =
    mhrpass0(t, comp_mixed, y1, y2, o2mixed, cnt, adj, zero(adj))
@inline mhrpass0(t, comp_mixed, y1, o2mixed, cnt, adj) =
    mhrpass0(t, comp_mixed, y1, nothing, o2mixed, cnt, adj, zero(adj))
@inline mhrpass0(t, ::Nothing, pairs_seq::AbstractVector, o2mixed::Int, cnt::Int, adj) =
    mhrpass0(t, nothing, nothing, pairs_seq, o2mixed, cnt, adj, zero(adj))

@inline function mhrpass0(t::SecondAdjointNode1, comp_mixed, y1, y2, o2mixed, cnt, adj, adj2)
    mhrpass0(t.inner, comp_mixed, y1, y2, o2mixed, cnt,
        adj * t.y,
        adj2 * (t.y)^2 + adj * t.h,
    )
end

@inline function mhrpass0(t::SecondAdjointNode2, comp_mixed, y1, y2, o2mixed, cnt, adj, adj2)
    adj2y1y2 = adj2 * t.y1 * t.y2
    adjh12 = adj * t.h12
    cnt = mhrpass0(t.inner1, comp_mixed, y1, y2, o2mixed, cnt,
        adj * t.y1,
        adj2 * (t.y1)^2 + adj * t.h11,
    )
    cnt = mhrpass0(t.inner2, comp_mixed, y1, y2, o2mixed, cnt,
        adj * t.y2,
        adj2 * (t.y2)^2 + adj * t.h22,
    )
    cnt = mhdrpass(t.inner1, t.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj2y1y2 + adjh12)
    cnt
end

@inline function mhrpass0(
    t::SecondAdjointNode2{typeof(+)},
    comp_mixed, y1, y2, o2mixed, cnt, adj, adj2,
)
    cnt = mhrpass0(t.inner1, comp_mixed, y1, y2, o2mixed, cnt, adj, adj2)
    cnt = mhrpass0(t.inner2, comp_mixed, y1, y2, o2mixed, cnt, adj, adj2)
    cnt
end

@inline function mhrpass0(
    t::SecondAdjointNode2{typeof(-)},
    comp_mixed, y1, y2, o2mixed, cnt, adj, adj2,
)
    cnt = mhrpass0(t.inner1, comp_mixed, y1, y2, o2mixed, cnt, adj, adj2)
    cnt = mhrpass0(t.inner2, comp_mixed, y1, y2, o2mixed, cnt, -adj, adj2)
    cnt
end

@inline mhrpass0(
    t::Union{SecondAdjointNodeVar,SecondAdjointParameterNode,SecondAdjointNull,Real},
    comp_mixed, y1, y2, o2mixed, cnt, adj, adj2) = cnt

@inline function mhrpass0(
    t::T, comp, y1::Tuple{V1,V2}, o2mixed, cnt, adj, adj2,
) where {T<:Union{SecondAdjointNull,Real,SecondAdjointParameterNode,SecondAdjointNodeVar},V1<:AbstractVector,V2<:AbstractVector}
    cnt
end

@inline function mhrpass0(
    t::SecondAdjointNode1, comp, y1::Tuple{V1,V2}, o2mixed, cnt, adj, adj2,
) where {V1<:AbstractVector,V2<:AbstractVector}
    mhrpass0(t.inner, comp, y1, o2mixed, cnt, adj * t.y, adj2 * (t.y)^2 + adj * t.h)
end

@inline function mhrpass0(
    t::SecondAdjointNode2, comp, y1::Tuple{V1,V2}, o2mixed, cnt, adj, adj2,
) where {V1<:AbstractVector,V2<:AbstractVector}
    adj2y1y2 = adj2 * t.y1 * t.y2
    adjh12 = adj * t.h12
    cnt = mhrpass0(t.inner1, comp, y1, o2mixed, cnt, adj * t.y1, adj2 * (t.y1)^2 + adj * t.h11)
    cnt = mhrpass0(t.inner2, comp, y1, o2mixed, cnt, adj * t.y2, adj2 * (t.y2)^2 + adj * t.h22)
    cnt = mhdrpass(t.inner1, t.inner2, comp, y1, nothing, o2mixed, cnt, adj2y1y2 + adjh12)
    cnt
end

@inline function mhrpass0(
    t::SecondAdjointNode2{typeof(+)}, comp, y1::Tuple{V1,V2}, o2mixed, cnt, adj, adj2,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt = mhrpass0(t.inner1, comp, y1, o2mixed, cnt, adj, adj2)
    cnt = mhrpass0(t.inner2, comp, y1, o2mixed, cnt, adj, adj2)
    cnt
end

@inline function mhrpass0(
    t::SecondAdjointNode2{typeof(-)}, comp, y1::Tuple{V1,V2}, o2mixed, cnt, adj, adj2,
) where {V1<:AbstractVector,V2<:AbstractVector}
    cnt = mhrpass0(t.inner1, comp, y1, o2mixed, cnt, adj, adj2)
    cnt = mhrpass0(t.inner2, comp, y1, o2mixed, cnt, -adj, adj2)
    cnt
end

@inline _mixed_adj(adj, f, k) = adj
@inline _mixed_adj(adj::AbstractVector, f, k) = @inbounds adj[offset0(f, k)]

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
            _mixed_adj(adj, f, k),
        )
    end
end

function smhessian!(y1::Nothing, y2::Tuple{VI,VI}, f, p, x, θ, comp, o2mixed, adj) where {I<:Integer,VI<:AbstractVector{I}}
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    rows, cols = y2
    mhrpass0(graph, comp, rows, cols, o2mixed, 0, adj)
end
function smhessian!(
    y1::Tuple{V1,V2},
    y2::Nothing,
    f, p, x, θ, comp, o2mixed, adj,
) where {V1<:AbstractVector,V2<:AbstractVector}
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    # Route tuple products through the dedicated tuple recursion.
    # The generic (y1, y2=nothing) recursion over-scales mixed products.
    mhrpass0(graph, comp, y1, o2mixed, 0, adj)
end
function smhessian!(y1::AbstractVector, y2::Nothing, f, p, x, θ, comp, o2mixed, adj)
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    mhrpass0(graph, comp, y1, o2mixed, 0, adj)
end
function smhessian!(y1, y2, f, p, x, θ, comp, o2mixed, adj)
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    mhrpass0(graph, comp, y1, y2, o2mixed, 0, adj)
end
