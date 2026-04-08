@inline function mhdrpass(
    t1::SecondAdjointNodeVar,
    t2::SecondAdjointParameterNode,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    hdrpass(t1, t2, comp, y1, y2, o2, cnt, adj)
end

@inline function mhdrpass(
    t1::SecondAdjointParameterNode,
    t2::SecondAdjointNodeVar,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    hdrpass(t1, t2, comp, y1, y2, o2, cnt, adj)
end

@inline mhdrpass(::Any, ::Any, comp, y1, y2, o2, cnt, adj) = cnt

@inline function mhdrpass(
    t1,
    t2::SecondAdjointNode1,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    mhdrpass(t1, t2.inner, comp, y1, y2, o2, cnt, adj * t2.y)
end

@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    mhdrpass(t1.inner, t2, comp, y1, y2, o2, cnt, adj * t1.y)
end

@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode1,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    mhdrpass(t1.inner, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y * t2.y)
end

@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    cnt = mhdrpass(t1.inner1, t2, comp, y1, y2, o2, cnt, adj * t1.y1)
    mhdrpass(t1.inner2, t2, comp, y1, y2, o2, cnt, adj * t1.y2)
end

@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode1,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    cnt = mhdrpass(t1.inner1, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y)
    mhdrpass(t1.inner2, t2.inner, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y)
end

@inline function mhdrpass(
    t1,
    t2::SecondAdjointNode2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    cnt = mhdrpass(t1, t2.inner1, comp, y1, y2, o2, cnt, adj * t2.y1)
    mhdrpass(t1, t2.inner2, comp, y1, y2, o2, cnt, adj * t2.y2)
end

@inline function mhdrpass(
    t1::SecondAdjointNode1,
    t2::SecondAdjointNode2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    cnt = mhdrpass(t1.inner, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y * t2.y1)
    mhdrpass(t1.inner, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y * t2.y2)
end

@inline function mhdrpass(
    t1::SecondAdjointNode2,
    t2::SecondAdjointNode2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
)
    cnt = mhdrpass(t1.inner1, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y1)
    cnt = mhdrpass(t1.inner1, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y1 * t2.y2)
    cnt = mhdrpass(t1.inner2, t2.inner1, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y1)
    mhdrpass(t1.inner2, t2.inner2, comp, y1, y2, o2, cnt, adj * t1.y2 * t2.y2)
end

@inline function mhrpass0(
    t::Union{SecondAdjointLeaf,SecondAdjointNull,Real},
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
)
    cnt
end

@inline function mhrpass0(
    t::SecondAdjointNode1,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
)
    mhrpass0(t.inner, comp, y1, y2, o2, cnt, adj * t.y, adj2 * (t.y)^2 + adj * t.h)
end

@inline function mhrpass0(
    t::SecondAdjointNode2,
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
)
    adj2y1y2 = adj2 * t.y1 * t.y2
    adjh12 = adj * t.h12
    cnt = mhrpass0(t.inner1, comp, y1, y2, o2, cnt, adj * t.y1, adj2 * (t.y1)^2 + adj * t.h11)
    cnt = mhrpass0(t.inner2, comp, y1, y2, o2, cnt, adj * t.y2, adj2 * (t.y2)^2 + adj * t.h22)
    mhdrpass(t.inner1, t.inner2, comp, y1, y2, o2, cnt, adj2y1y2 + adjh12)
end

@inline function mhrpass0(
    t::SecondAdjointNode2{typeof(+)},
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
)
    cnt = mhrpass0(t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = mhrpass0(t.inner2, comp, y1, y2, o2, cnt, adj, adj2)
    cnt
end

@inline function mhrpass0(
    t::SecondAdjointNode2{typeof(-)},
    comp,
    y1,
    y2,
    o2,
    cnt,
    adj,
    adj2,
)
    cnt = mhrpass0(t.inner1, comp, y1, y2, o2, cnt, adj, adj2)
    cnt = mhrpass0(t.inner2, comp, y1, y2, o2, cnt, -adj, adj2)
    cnt
end

@inline _hess_param_adj(adj, f, k) = adj
@inline _hess_param_adj(adj::AbstractVector, f, k) = @inbounds adj[offset0(f, k)]

function shessian_param!(y1, y2, f, x, θ, adj)
    @simd for k in eachindex(f.itr)
        @inbounds shessian_param!(
            y1,
            y2,
            f.f,
            f.itr[k],
            x,
            θ,
            f.f.mcomp2,
            offset2_param(f, k),
            _hess_param_adj(adj, f, k),
        )
    end
end

function shessian_param!(y1, y2, f, p, x, θ, comp, o2, adj)
    graph = f(p, SecondAdjointNodeSource(x), SecondAdjointParameterSource(θ))
    mhrpass0(graph, comp, y1, y2, o2, 0, adj, zero(adj))
end
