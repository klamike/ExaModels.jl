using ExaModels
using NLPModels
using ParametricNLPModels
using SparseArrays
using Test

const FD_EPS = 1e-7

function _set_theta!(c::ExaCore, θhandle, θval::AbstractVector)
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval)
    else
        copyto!(c.θ, θval)
    end
    return nothing
end

function _model_and_nlp(c::ExaCore; prod::Bool = true)
    m = ExaModel(c; prod=prod)
    return m, WrapperNLPModel(m)
end

function _basis_matrix(apply!, nout::Int, nin::Int)
    A = zeros(nout, nin)
    v = zeros(nin)
    out = zeros(nout)
    for j in 1:nin
        fill!(v, 0.0)
        v[j] = 1.0
        fill!(out, 0.0)
        apply!(out, v)
        @views A[:, j] .= out
    end
    return A
end

function _jpprod_matrix(m::ExaModel, nlp::WrapperNLPModel, x::AbstractVector)
    _basis_matrix(NLPModels.get_ncon(m), get_nparam(m)) do out, v
        ExaModels.jpprod!(nlp, x, v, out)
    end
end

function _jptprod_matrix(m::ExaModel, nlp::WrapperNLPModel, x::AbstractVector)
    _basis_matrix(get_nparam(m), NLPModels.get_ncon(m)) do out, v
        ExaModels.jptprod!(nlp, x, v, out)
    end
end

function _hpprod_matrix(
    m::ExaModel,
    nlp::WrapperNLPModel,
    x::AbstractVector;
    y::Union{Nothing,AbstractVector} = nothing,
    obj_weight::Real = 1.0,
)
    _basis_matrix(NLPModels.get_nvar(m), get_nparam(m)) do out, v
        if isnothing(y)
            ExaModels.hpprod!(nlp, x, v, out; obj_weight=obj_weight)
        else
            ExaModels.hpprod!(nlp, x, y, v, out; obj_weight=obj_weight)
        end
    end
end

function _hptprod_matrix(
    m::ExaModel,
    nlp::WrapperNLPModel,
    x::AbstractVector;
    y::Union{Nothing,AbstractVector} = nothing,
    obj_weight::Real = 1.0,
)
    _basis_matrix(get_nparam(m), NLPModels.get_nvar(m)) do out, v
        if isnothing(y)
            ExaModels.hptprod!(nlp, x, v, out; obj_weight=obj_weight)
        else
            ExaModels.hptprod!(nlp, x, y, v, out; obj_weight=obj_weight)
        end
    end
end

function _jac_param_from_coord(m::ExaModel, nlp::WrapperNLPModel, x::AbstractVector)
    rows = zeros(Int, get_nnzjp(m))
    cols = zeros(Int, get_nnzjp(m))
    vals = zeros(get_nnzjp(m))
    ExaModels.jac_param_structure!(nlp, rows, cols)
    ExaModels.jac_param_coord!(nlp, x, vals)
    return sparse(rows, cols, vals, NLPModels.get_ncon(m), get_nparam(m))
end

function _hess_param_from_coord(m::ExaModel, x::AbstractVector; y::Union{Nothing,AbstractVector} = nothing, obj_weight::Float64 = 1.0)
    rows = zeros(Int, get_nnzhp(m))
    cols = zeros(Int, get_nnzhp(m))
    vals = zeros(get_nnzhp(m))
    ExaModels.hess_param_structure!(m, rows, cols)
    if isnothing(y)
        ExaModels.hess_param_coord!(m, x, vals; obj_weight=obj_weight)
    else
        ExaModels.hess_param_coord!(m, x, y, vals; obj_weight=obj_weight)
    end
    return sparse(rows, cols, vals, NLPModels.get_nvar(m), get_nparam(m))
end

function _lagrangian_grad_x!(nlp::WrapperNLPModel,
    x::AbstractVector, y::AbstractVector, gradL::AbstractVector, work::AbstractVector;
    obj_weight::Float64 = 1.0,
)
    fill!(work, 0.0)
    NLPModels.grad!(nlp, x, work)
    @. gradL = obj_weight * work
    if !isempty(y)
        fill!(work, 0.0)
        NLPModels.jtprod_nln!(nlp, x, y, work)
        @. gradL += work
    end
    return gradL
end

function _fd_grad_param(
    nlp::WrapperNLPModel,
    c::ExaCore,
    θhandle,
    θval::AbstractVector,
    x::AbstractVector;
    ε = FD_EPS,
)
    g = zeros(length(θval))
    _set_theta!(c, θhandle, θval)
    obj0 = NLPModels.obj(nlp, x)
    for j in eachindex(g)
        θp = copy(θval)
        θp[j] += ε
        _set_theta!(c, θhandle, θp)
        g[j] = (NLPModels.obj(nlp, x) - obj0) / ε
    end
    _set_theta!(c, θhandle, θval)
    return g
end

function _fd_hpprod!(nlp::WrapperNLPModel, c::ExaCore,
    θhandle, θval::AbstractVector, x::AbstractVector, y::AbstractVector, vparam::AbstractVector, out::AbstractVector;
    obj_weight = 1.0, ε = FD_EPS,
)
    gradL = similar(out)
    gradL_pert = similar(out)
    work = similar(out)

    _set_theta!(c, θhandle, θval)
    _lagrangian_grad_x!(nlp, x, y, gradL, work; obj_weight=obj_weight)

    _set_theta!(c, θhandle, θval .+ ε .* vparam)
    _lagrangian_grad_x!(nlp, x, y, gradL_pert, work; obj_weight=obj_weight)

    _set_theta!(c, θhandle, θval)
    @. out = (gradL_pert - gradL) / ε
    return out
end

function _fd_hptprod!(nlp::WrapperNLPModel, c::ExaCore,
    θhandle, θval::AbstractVector, x::AbstractVector, y::AbstractVector, uvar::AbstractVector, out::AbstractVector;
    obj_weight = 1.0, ε = FD_EPS,
)
    gradL = similar(uvar)
    work = similar(uvar)

    φ = function ()
        _lagrangian_grad_x!(nlp, x, y, gradL, work; obj_weight=obj_weight)
        return sum(gradL .* uvar)
    end

    _set_theta!(c, θhandle, θval)
    φ0 = φ()

    for j in eachindex(out)
        θp = copy(θval)
        θp[j] += ε
        _set_theta!(c, θhandle, θp)
        out[j] = (φ() - φ0) / ε
    end

    _set_theta!(c, θhandle, θval)
    return out
end

function _check_hprod_pair_with_fd!(nlp::WrapperNLPModel, c::ExaCore,
    θ, θval::AbstractVector, x::AbstractVector, y::AbstractVector, v_param::AbstractVector, v_var::AbstractVector;
    obj_weight::Float64 = 1.0,
)
    Hv = zeros(length(v_var))
    HTu = zeros(length(v_param))
    ExaModels.hpprod!(nlp, x, y, v_param, Hv; obj_weight=obj_weight)
    ExaModels.hptprod!(nlp, x, y, v_var, HTu; obj_weight=obj_weight)

    @test sum(Hv .* v_var) ≈ sum(v_param .* HTu) atol=1e-12

    fd_Hv = zeros(length(v_var))
    _fd_hpprod!(nlp, c, θ, θval, x, y, v_param, fd_Hv; obj_weight=obj_weight)
    @test Hv ≈ fd_Hv atol=1e-4

    fd_HTu = zeros(length(v_param))
    _fd_hptprod!(nlp, c, θ, θval, x, y, v_var, fd_HTu; obj_weight=obj_weight)
    @test HTu ≈ fd_HTu atol=1e-4

    return Hv, HTu
end

function test_parameter_derivatives(backend)
    @testset "grad_param!" begin
        c = ExaCore(; backend=backend)
        x = variable(c, 3)
        θ = parameter(c, [1.0, 2.0, 3.0])
        objective(c, θ[i] * x[i]^2 for i in 1:3)

        m = ExaModel(c)
        g = ExaModels.convert_array(zeros(3), backend)
        x1 = ExaModels.convert_array([1.0, 2.0, 3.0], backend)
        x2 = ExaModels.convert_array([2.0, 3.0, 4.0], backend)

        ExaModels.grad_param!(m, x1, g)
        @test Array(g) ≈ [1.0, 4.0, 9.0] atol=1e-12

        fill!(g, 0.0)
        ExaModels.grad_param!(m, x2, g)
        @test Array(g) ≈ [4.0, 9.0, 16.0] atol=1e-12

        set_parameter!(c, θ, [10.0, 20.0, 30.0])
        fill!(g, 0.0)
        ExaModels.grad_param!(m, x1, g)
        @test Array(g) ≈ [1.0, 4.0, 9.0] atol=1e-12

        set_parameter!(c, θ, [1.0, 2.0, 3.0])
        nlp = WrapperNLPModel(m)
        x_cpu = [1.0, 2.0, 3.0]
        g_fd = _fd_grad_param(nlp, c, θ, [1.0, 2.0, 3.0], x_cpu)
        fill!(g, 0.0)
        ExaModels.grad_param!(m, x1, g)
        @test Array(g) ≈ g_fd atol=1e-4

        c2 = ExaCore(; backend=backend)
        x2v = variable(c2, 2)
        θ2 = parameter(c2, [3.0, 5.0])
        objective(c2, θ2[1] * x2v[1]^2)
        constraint(c2, θ2[1] * x2v[1] + θ2[2] * x2v[2] - 1.0)

        m2 = ExaModel(c2)
        g2 = ExaModels.convert_array(zeros(2), backend)
        ExaModels.grad_param!(m2, ExaModels.convert_array([2.0, 3.0], backend), g2)
        @test Array(g2) ≈ [4.0, 0.0] atol=1e-12
    end

    @testset "jac_param!" begin
        c = ExaCore(; backend=backend)
        x = variable(c, 2)
        θ = parameter(c, [1.0, 2.0])
        constraint(c, θ[1] * x[1] + θ[2] * x[2])
        constraint(c, θ[1] * x[1]^2 + θ[2] * x[2]^2)
        m, nlp = _model_and_nlp(c; prod=true)
        x_test = [1.0, 2.0]
        expected_J = [1.0 2.0; 1.0 4.0]

        Jp = _jpprod_matrix(m, nlp, x_test)
        JpT = _jptprod_matrix(m, nlp, x_test)

        @test Jp ≈ expected_J atol=1e-12
        @test JpT ≈ expected_J' atol=1e-12
        @test Jp ≈ JpT' atol=1e-12

        for θv in ([1.0, 2.0], [3.0, 4.0], [0.5, 1.5])
            set_parameter!(c, θ, θv)
            JpTθ = _jptprod_matrix(m, nlp, x_test)
            @test JpTθ ≈ [x_test[1] x_test[1]^2; x_test[2] x_test[2]^2] atol=1e-12
        end

        csp = ExaCore(; backend=backend)
        xsp = variable(csp, 3)
        θsp = parameter(csp, [1.0, 2.0, 3.0])
        constraint(csp, θsp[1] * xsp[1] + θsp[2] * xsp[2])
        constraint(csp, θsp[2] * xsp[2] + θsp[3] * xsp[3])
        constraint(csp, θsp[1] * xsp[1] + θsp[3] * xsp[3])
        msp, nlpsp = _model_and_nlp(csp; prod=true)
        JspT = _jptprod_matrix(msp, nlpsp, [1.0, 2.0, 3.0])
        @test (abs.(JspT) .> 1e-12) == [1 0 1; 1 1 0; 0 1 1]

        ccoord = ExaCore(; backend=backend)
        xcoord = variable(ccoord, 4)
        θcoord = parameter(ccoord, [1.0, 2.0, 3.0])
        constraint(ccoord, θcoord[1] * xcoord[1]^2 + θcoord[3] * xcoord[3])
        constraint(ccoord, θcoord[2] * xcoord[2] + θcoord[3] * xcoord[4]^2)
        constraint(ccoord, θcoord[1] * xcoord[1] + θcoord[2] * xcoord[3] + θcoord[3] * xcoord[4])
        mcoord, nlpcoord = _model_and_nlp(ccoord; prod=true)
        x_coord = [1.0, 2.0, 3.0, 4.0]
        expected_J = [
            1.0 0.0 3.0
            0.0 2.0 16.0
            1.0 3.0 4.0
        ]

        J_prod = _jpprod_matrix(mcoord, nlpcoord, x_coord)
        J_tprod = _jptprod_matrix(mcoord, nlpcoord, x_coord)
        J_coord = Matrix(_jac_param_from_coord(mcoord, nlpcoord, x_coord))

        @test J_prod ≈ expected_J atol=1e-12
        @test J_tprod ≈ expected_J' atol=1e-12
        @test J_coord ≈ expected_J atol=1e-12
    end

    @testset "hess_param!" begin
        cs = ExaCore(; backend=backend)
        xs = variable(cs, 3)
        θs = parameter(cs, [1.0, 2.0, 3.0])
        objective(cs, θs[1] * xs[1]^2 + θs[2] * xs[1] * xs[2] + θs[3] * xs[2] * xs[3])
        ms, nlps = _model_and_nlp(cs; prod=true)
        x_s = [1.0, 2.0, 3.0]
        expected_Hs = [
            2.0 2.0 0.0
            0.0 1.0 3.0
            0.0 0.0 2.0
        ]

        Hs = _hpprod_matrix(ms, nlps, x_s; y=zeros(0), obj_weight=1.0)
        @test (abs.(Hs) .> 1e-12) == [1 1 0; 0 1 1; 0 0 1]
        @test Hs ≈ expected_Hs atol=1e-12

        for θv in ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [0.5, 1.5, 2.5])
            set_parameter!(cs, θs, θv)
            Hθ = _hpprod_matrix(ms, nlps, x_s; y=zeros(0), obj_weight=1.0)
            @test Hθ ≈ expected_Hs atol=1e-12
        end

        c1 = ExaCore(; backend=backend)
        x1 = variable(c1, 2)
        θ1 = parameter(c1, [1.0, 2.0])
        objective(c1, θ1[1] * x1[1]^2 + θ1[2] * x1[1] * x1[2])
        constraint(c1, θ1[1] * x1[1] + θ1[2] * x1[2] - 3.0)
        _, nlp1 = _model_and_nlp(c1; prod=true)

        r0 = zeros(2)
        ExaModels.hpprod!(nlp1, [1.0, 2.0], [0.0], [1.0, 0.5], r0; obj_weight=1.0)
        @test r0 ≈ [3.0, 0.5] atol=1e-12

        r1 = zeros(2)
        ExaModels.hpprod!(nlp1, [1.0, 2.0], [1.0], [1.0, 0.5], r1; obj_weight=1.0)
        @test r1 ≈ [4.0, 1.0] atol=1e-12

        fd0 = zeros(2)
        _fd_hpprod!(nlp1, c1, θ1, [1.0, 2.0], [1.0, 2.0], zeros(0), [1.0, 0.5], fd0; obj_weight=1.0)
        @test r0 ≈ fd0 atol=1e-4

        fd1 = zeros(2)
        _fd_hpprod!(nlp1, c1, θ1, [1.0, 2.0], [1.0, 2.0], [1.0], [1.0, 0.5], fd1; obj_weight=1.0)
        @test r1 ≈ fd1 atol=1e-4

        c2 = ExaCore(; backend=backend)
        x2 = variable(c2, 2)
        θ2 = parameter(c2, [0.5, 1.5])
        objective(c2, θ2[1] * x2[1]^3 + θ2[2] * x2[1]^2 * x2[2] + θ2[1] * θ2[2] * x2[2]^2)
        _, nlp2 = _model_and_nlp(c2; prod=true)
        _check_hprod_pair_with_fd!(
            nlp2,
            c2,
            θ2,
            [0.5, 1.5],
            [1.5, 2.5],
            zeros(0),
            [2.0, 1.0],
            [1.0, 0.5];
            obj_weight=1.0,
        )

        c3 = ExaCore(; backend=backend)
        x3 = variable(c3, 3)
        θ3 = parameter(c3, [2.0, 1.0, 0.5])
        objective(c3, θ3[1] * x3[1]^2 + θ3[2] * x3[2]^2 + θ3[3] * x3[1] * x3[2])
        constraint(c3, θ3[1] * x3[1] + θ3[2] * x3[2] + x3[3] - 1.0)
        constraint(c3, x3[1]^2 + θ3[3] * x3[2]^2 - θ3[1])
        _, nlp3 = _model_and_nlp(c3; prod=true)

        Hv3, _ = _check_hprod_pair_with_fd!(
            nlp3,
            c3,
            θ3,
            [2.0, 1.0, 0.5],
            [0.5, 1.0, 0.2],
            [0.5, 1.5],
            [1.0, 0.5, 2.0],
            [1.0, 0.5, 0.25];
            obj_weight=1.0,
        )
        Hv3b = zeros(3)
        ExaModels.hpprod!(nlp3, [0.5, 1.0, 0.2], [1.0, 0.5], [1.0, 0.5, 2.0], Hv3b; obj_weight=1.0)
        @test !(Hv3 ≈ Hv3b)

        cobj = ExaCore(; backend=backend)
        xobj = variable(cobj, 3)
        θobj = parameter(cobj, [1.0, 2.0])
        objective(cobj, θobj[1] * xobj[1]^2 + θobj[2] * xobj[2]^2 + θobj[1] * θobj[2] * xobj[3]^2)
        constraint(cobj, θobj[1] * xobj[1] + θobj[2] * xobj[2] - 1.0)
        mobj, nlpobj = _model_and_nlp(cobj; prod=true)
        x_obj = [1.0, 2.0, 3.0]

        Hpt_obj = zeros(2)
        ExaModels.hptprod!(nlpobj, x_obj, [1.0, 1.0, 1.0], Hpt_obj)
        @test Hpt_obj ≈ [14.0, 10.0] atol=1e-12
        ExaModels.hptprod!(nlpobj, x_obj, [1.0], [1.0, 1.0, 1.0], Hpt_obj)
        @test Hpt_obj ≈ [15.0, 11.0] atol=1e-12

        H_obj = zeros(3)
        ExaModels.hpprod!(nlpobj, x_obj, [1.0, 1.0], H_obj)
        @test H_obj ≈ [2.0, 4.0, 12.0 + 6.0] atol=1e-12

        H_obj_lag = zeros(3)
        ExaModels.hpprod!(nlpobj, x_obj, [1.0], [1.0, 1.0], H_obj_lag)
        @test H_obj_lag ≈ [3.0, 5.0, 18.0] atol=1e-12

        H_coord_obj = Matrix(_hess_param_from_coord(mobj, x_obj; obj_weight=1.0))
        @test H_coord_obj ≈ [2.0 0.0; 0.0 4.0; 12.0 6.0] atol=1e-12

        ccoord = ExaCore(; backend=backend)
        xcoord = variable(ccoord, 3)
        θcoord = parameter(ccoord, [1.0, 2.0])
        objective(ccoord, θcoord[1] * xcoord[1]^2 + θcoord[2] * xcoord[2]^2 + θcoord[1] * θcoord[2] * xcoord[3]^2)
        constraint(ccoord, θcoord[1] * xcoord[1] + θcoord[2] * xcoord[2] - 5.0)
        mcoord, nlpcoord = _model_and_nlp(ccoord; prod=true)

        x_m = [1.0, 2.0, 3.0]
        y_m = [0.5]
        H_prod = _hpprod_matrix(mcoord, nlpcoord, x_m; y=y_m, obj_weight=1.0)
        H_tprod = _hptprod_matrix(mcoord, nlpcoord, x_m; y=y_m, obj_weight=1.0)
        H_coord = Matrix(_hess_param_from_coord(mcoord, x_m; y=y_m, obj_weight=1.0))

        @test H_prod ≈ H_tprod' atol=1e-12
        @test H_coord ≈ H_prod atol=1e-12
        @test H_prod ≈ [2.5 0.0; 0.0 4.5; 12.0 6.0] atol=1e-12

        csparse = ExaCore(; backend=backend)
        xsparse = variable(csparse, 5)
        θsparse = parameter(csparse, [1.0, 2.0, 3.0, 4.0])
        objective(csparse, sum(θsparse[i] * xsparse[i]^2 for i in 1:4))
        constraint(csparse, θsparse[1] * xsparse[1] + θsparse[2] * xsparse[2])
        constraint(csparse, θsparse[2] * xsparse[2] + θsparse[3] * xsparse[3])
        constraint(csparse, θsparse[3] * xsparse[3] + θsparse[4] * xsparse[4])
        constraint(csparse, θsparse[1] * xsparse[1] + θsparse[4] * xsparse[5])
        msparse, nlpsparse = _model_and_nlp(csparse; prod=true)

        x_s = ones(5)
        y_s = [0.1, 0.2, 0.3, 0.4]

        J_sparse = _jpprod_matrix(msparse, nlpsparse, x_s)
        @test nnz(sparse(J_sparse)) == 8

        H_sparse = _hpprod_matrix(msparse, nlpsparse, x_s; y=y_s, obj_weight=1.0)
        @test nnz(sparse(H_sparse)) == 5

        HT_sparse = _hptprod_matrix(msparse, nlpsparse, x_s; y=y_s, obj_weight=1.0)
        @test H_sparse ≈ HT_sparse' atol=1e-12

        H_sparse_coord = _hess_param_from_coord(msparse, x_s; y=y_s, obj_weight=1.0)
        @test nnz(H_sparse_coord) == 5
        @test Matrix(H_sparse_coord) ≈ H_sparse atol=1e-12
    end
end
