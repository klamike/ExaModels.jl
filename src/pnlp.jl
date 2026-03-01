function _parametric_meta(
    c::ExaCore;
    grad_param_available::Bool = true,
    jac_param_available::Bool = true,
    hess_param_available::Bool = true,
    jpprod_available::Bool = true,
    jptprod_available::Bool = true,
    hpprod_available::Bool = true,
    hptprod_available::Bool = true,
)
    return NLPModels.ParametricNLPModelMeta(
        nparam               = length(c.θ),
        nnzjp                = c.nnzjp,
        nnzhp                = c.nnzmh,
        grad_param_available = grad_param_available,
        jac_param_available  = jac_param_available,
        hess_param_available = hess_param_available,
        jpprod_available     = jpprod_available,
        jptprod_available    = jptprod_available,
        hpprod_available     = hpprod_available,
        hptprod_available    = hptprod_available,
    )
end

function hess_param_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)
    _obj_hess_param_structure!(m.objs, rows, cols)
    _con_hess_param_structure!(m.cons, rows, cols)
    return rows, cols
end

_obj_hess_param_structure!(objs::ObjectiveNull, rows, cols) = nothing
function _obj_hess_param_structure!(objs, rows, cols)
    _obj_hess_param_structure!(objs.inner, rows, cols)
    if objs.f.mo2step > 0
        smhessian!(rows, cols, objs, nothing, nothing, one(Float64))
    end
end

_con_hess_param_structure!(cons::ConstraintNull, rows, cols) = nothing
function _con_hess_param_structure!(cons, rows, cols)
    _con_hess_param_structure!(cons.inner, rows, cols)
    if cons.f.mo2step > 0
        smhessian!(rows, cols, cons, nothing, nothing, NaN)
    end
end

function grad_param!(m::ExaModel, x::AbstractVector, g::AbstractVector)
    fill!(g, zero(eltype(g)))
    _grad_param!(m.objs, x, m.θ, g)
    return g
end

function _grad_param!(objs, x, θ, g)
    _grad_param!(objs.inner, x, θ, g)
    grad_param!(g, objs, x, θ, one(eltype(g)))
end
_grad_param!(objs::ObjectiveNull, x, θ, g) = nothing

function jac_param_structure!(m::ExaModel, rows::AbstractVector, cols::AbstractVector)
    _jac_param_structure!(m.cons, rows, cols)
    return rows, cols
end

_jac_param_structure!(cons::ConstraintNull, rows, cols) = nothing
function _jac_param_structure!(cons, rows, cols)
    _jac_param_structure!(cons.inner, rows, cols)
    if cons.f.po1step > 0
        sjacobianp!(rows, cols, cons, nothing, nothing, NaN)
    end
end

function jac_param_coord!(m::ExaModel, x::AbstractVector, jac::AbstractVector)
    fill!(jac, zero(eltype(jac)))
    _jac_param_coord!(m.cons, x, m.θ, jac)
    return jac
end

_jac_param_coord!(cons::ConstraintNull, x, θ, jac) = nothing
function _jac_param_coord!(cons, x, θ, jac)
    _jac_param_coord!(cons.inner, x, θ, jac)
    if cons.f.po1step > 0
        sjacobianp!(jac, nothing, cons, x, θ, one(eltype(jac)))
    end
end

function jpprod!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jpv::AbstractVector)
    fill!(Jpv, zero(eltype(Jpv)))
    _jpprod!(m.cons, x, m.θ, v, Jpv)
    return Jpv
end

_jpprod!(cons::ConstraintNull, x, θ, v, Jpv) = nothing
function _jpprod!(cons, x, θ, v, Jpv)
    _jpprod!(cons.inner, x, θ, v, Jpv)
    if cons.f.po1step > 0
        sjacobianp!((Jpv, v), nothing, cons, x, θ, one(eltype(Jpv)))
    end
end

function jptprod!(m::ExaModel, x::AbstractVector, v::AbstractVector, Jptv::AbstractVector)
    fill!(Jptv, zero(eltype(Jptv)))
    _jptprod!(m.cons, x, m.θ, v, Jptv)
    return Jptv
end

_jptprod!(cons::ConstraintNull, x, θ, v, Jptv) = nothing
function _jptprod!(cons, x, θ, v, Jptv)
    _jptprod!(cons.inner, x, θ, v, Jptv)
    if cons.f.po1step > 0
        sjacobianp!(nothing, (Jptv, v), cons, x, θ, one(eltype(Jptv)))
    end
end

function hess_param_coord!(
    m::ExaModel,
    x::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(hess, zero(eltype(hess)))
    _obj_hess_param_coord!(m.objs, x, m.θ, hess, obj_weight)
    return hess
end

function hess_param_coord!(
    m::ExaModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight = one(eltype(x)),
)
    fill!(hess, zero(eltype(hess)))
    _obj_hess_param_coord!(m.objs, x, m.θ, hess, obj_weight)
    _con_hess_param_coord!(m.cons, x, m.θ, y, hess, obj_weight)
    return hess
end

_obj_hess_param_coord!(objs::ObjectiveNull, x, θ, hess, obj_weight) = nothing
function _obj_hess_param_coord!(objs, x, θ, hess, obj_weight)
    _obj_hess_param_coord!(objs.inner, x, θ, hess, obj_weight)
    if objs.f.mo2step > 0
        smhessian!(hess, nothing, objs, x, θ, obj_weight)
    end
end

_con_hess_param_coord!(cons::ConstraintNull, x, θ, y, hess, obj_weight) = nothing
function _con_hess_param_coord!(cons, x, θ, y, hess, obj_weight)
    _con_hess_param_coord!(cons.inner, x, θ, y, hess, obj_weight)
    if cons.f.mo2step > 0
        smhessian!(hess, nothing, cons, x, θ, y)
    end
end

function hptprod!(m::ExaModel, x::AbstractVector, v::AbstractVector, Hmtv::AbstractVector; obj_weight = one(eltype(x)))
    fill!(Hmtv, zero(eltype(Hmtv)))
    if m.pmeta.nnzhp == 0
        return Hmtv
    end
    _obj_hptprod!(m.objs, x, m.θ, v, Hmtv, obj_weight)
    return Hmtv
end

function hptprod!(m::ExaModel, x::AbstractVector, y::AbstractVector, v::AbstractVector, Hmtv::AbstractVector; obj_weight = one(eltype(x)))
    fill!(Hmtv, zero(eltype(Hmtv)))
    if m.pmeta.nnzhp == 0
        return Hmtv
    end
    _obj_hptprod!(m.objs, x, m.θ, v, Hmtv, obj_weight)
    _con_hptprod!(m.cons, x, m.θ, y, v, Hmtv, obj_weight)
    return Hmtv
end

function hpprod!(m::ExaModel, x::AbstractVector, v::AbstractVector, Hmv::AbstractVector; obj_weight = one(eltype(x)))
    fill!(Hmv, zero(eltype(Hmv)))
    if m.pmeta.nnzhp == 0
        return Hmv
    end
    _obj_hpprod!(m.objs, x, m.θ, v, Hmv, obj_weight)
    return Hmv
end

function hpprod!(m::ExaModel, x::AbstractVector, y::AbstractVector, v::AbstractVector, Hmv::AbstractVector; obj_weight = one(eltype(x)))
    fill!(Hmv, zero(eltype(Hmv)))
    if m.pmeta.nnzhp == 0
        return Hmv
    end
    _obj_hpprod!(m.objs, x, m.θ, v, Hmv, obj_weight)
    _con_hpprod!(m.cons, x, m.θ, y, v, Hmv, obj_weight)
    return Hmv
end

_obj_hptprod!(objs::ObjectiveNull, x, θ, v, Hmtv, obj_weight) = nothing
function _obj_hptprod!(objs, x, θ, v, Hmtv, obj_weight)
    _obj_hptprod!(objs.inner, x, θ, v, Hmtv, obj_weight)
    if objs.f.mo2step > 0
        smhessian!(nothing, (Hmtv, v), objs, x, θ, obj_weight)
    end
end

_con_hptprod!(cons::ConstraintNull, x, θ, y, v, Hmtv, obj_weight) = nothing
function _con_hptprod!(cons, x, θ, y, v, Hmtv, obj_weight)
    _con_hptprod!(cons.inner, x, θ, y, v, Hmtv, obj_weight)
    if cons.f.mo2step > 0
        smhessian!(nothing, (Hmtv, v), cons, x, θ, y)
    end
end

_obj_hpprod!(objs::ObjectiveNull, x, θ, v, Hmv, obj_weight) = nothing
function _obj_hpprod!(objs, x, θ, v, Hmv, obj_weight)
    _obj_hpprod!(objs.inner, x, θ, v, Hmv, obj_weight)
    if objs.f.mo2step > 0
        smhessian!((Hmv, v), nothing, objs, x, θ, obj_weight)
    end
end

_con_hpprod!(cons::ConstraintNull, x, θ, y, v, Hmv, obj_weight) = nothing
function _con_hpprod!(cons, x, θ, y, v, Hmv, obj_weight)
    _con_hpprod!(cons.inner, x, θ, y, v, Hmv, obj_weight)
    if cons.f.mo2step > 0
        smhessian!((Hmv, v), nothing, cons, x, θ, y)
    end
end
