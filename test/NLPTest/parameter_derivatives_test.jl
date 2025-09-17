using ExaModels
using NLPModels
using Test
using SparseArrays

# Finite-difference helpers for robust verification
const FD_EPS = 1e-7

function fd_param_jvp!(nlp::WrapperNLPModel, c::ExaCore, θhandle, θval::AbstractVector,
                       x::AbstractVector, vparam::AbstractVector, out::AbstractVector; ε=FD_EPS)
    # g(θ) = constraints(x; θ). Return (g(θ+εv) - g(θ))/ε
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval)
    else
        copyto!(c.θ, θval)
    end
    g0 = similar(out); fill!(g0, 0.0)
    NLPModels.cons_nln!(nlp, x, g0)
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval .+ ε .* vparam)
    else
        @. c.θ = θval + ε * vparam
    end
    g1 = similar(out); fill!(g1, 0.0)
    NLPModels.cons_nln!(nlp, x, g1)
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval)
    else
        copyto!(c.θ, θval)
    end
    @. out = (g1 - g0) / ε
    return out
end

function fd_mhprod!(nlp::WrapperNLPModel, c::ExaCore, θhandle, θval::AbstractVector,
                    x::AbstractVector, y::AbstractVector, vparam::AbstractVector,
                    out::AbstractVector; obj_weight=1.0, ε=FD_EPS)
    # H_mixed * v ≈ (∇_x L(θ+εv) - ∇_x L(θ))/ε, L = obj_weight*f + y' g
    gradL = similar(out)
    grad_obj = similar(out); tmp = similar(out)
    # θ
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval)
    else
        copyto!(c.θ, θval)
    end
    fill!(grad_obj, 0.0)
    NLPModels.grad!(nlp, x, grad_obj)
    @. gradL = obj_weight * grad_obj
    if length(y) > 0
        fill!(tmp, 0.0); NLPModels.jtprod_nln!(nlp, x, y, tmp)
        @. gradL += tmp
    end
    # θ+εv
    gradL_pert = similar(out)
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval .+ ε .* vparam)
    else
        @. c.θ = θval + ε * vparam
    end
    fill!(grad_obj, 0.0)
    NLPModels.grad!(nlp, x, grad_obj)
    @. gradL_pert = obj_weight * grad_obj
    if length(y) > 0
        fill!(tmp, 0.0); NLPModels.jtprod_nln!(nlp, x, y, tmp)
        @. gradL_pert += tmp
    end
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval)
    else
        copyto!(c.θ, θval)
    end
    @. out = (gradL_pert - gradL) / ε
    return out
end

function fd_mhtprod!(nlp::WrapperNLPModel, c::ExaCore, θhandle, θval::AbstractVector,
                     x::AbstractVector, y::AbstractVector, uvar::AbstractVector,
                     out::AbstractVector; obj_weight=1.0, ε=FD_EPS)
    # H_mixed' * u ≈ ∂/∂θ ( ∇_x L(θ) ⋅ u ) in direction of standard basis
    # We compute directional derivative along each θ basis using finite differences to assemble out
    # out[j] ≈ (φ(θ + ε e_j) - φ(θ)) / ε where φ(θ) = (∇_x L(θ))' u
    φ = function (θcurr)
        grad_obj = similar(uvar); NLPModels.grad!(nlp, x, grad_obj)
        jt = similar(uvar); fill!(jt, 0.0); length(y) > 0 && NLPModels.jtprod_nln!(nlp, x, y, jt)
        return obj_weight * sum(grad_obj .* uvar) + sum(jt .* uvar)
    end
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval)
    else
        copyto!(c.θ, θval)
    end
    φ0 = φ(θval)
    for j in eachindex(out)
        θp = copy(θval); θp[j] += ε
        if θhandle isa ExaModels.Parameter
            ExaModels.set_parameter!(c, θhandle, θp)
        else
            copyto!(c.θ, θp)
        end
        out[j] = (φ(θp) - φ0) / ε
    end
    if θhandle isa ExaModels.Parameter
        ExaModels.set_parameter!(c, θhandle, θval)
    else
        copyto!(c.θ, θval)
    end
    return out
end

"""
Test parameter jacobian consistency between jpprod! and jptprod!
"""
function test_parameter_jacobian_consistency(backend)
    c = ExaCore() 
    x = variable(c, 2)
    θ = parameter(c, [1.0, 2.0])
    
    constraint(c, θ[1] * x[1] + θ[2] * x[2])
    constraint(c, θ[1] * x[1]^2 + θ[2] * x[2]^2)
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [1.0, 2.0]
    
    # Test forward and transpose consistency
    v_param = [1.0, 0.5]
    v_constraint = [1.0, 0.5]
    
    # Forward: J_p * v_param (parameter space → constraint space)
    result_forward = zeros(2)
    ExaModels.jpprod!(nlp, x_test, v_param, result_forward)
    
    # Transpose: J_p^T * v_constraint (constraint space → parameter space)
    result_transpose = zeros(2)
    ExaModels.jptprod!(nlp, x_test, v_constraint, result_transpose)
    
    # Expected results:
    # J_p = [[1, 2], [1, 4]]  (at x_test = [1,2])
    # Forward: J_p * [1, 0.5] = [1*1 + 2*0.5, 1*1 + 4*0.5] = [2.0, 3.0]
    # Transpose: J_p^T * [1, 0.5] = [1*1 + 1*0.5, 2*1 + 4*0.5] = [1.5, 4.0]
    expected_forward = [2.0, 3.0]
    expected_transpose = [1.5, 4.0]
    
    @test result_forward ≈ expected_forward atol=1e-12
    @test result_transpose ≈ expected_transpose atol=1e-12
end

"""
Test parameter jacobian sparsity pattern
"""
function test_parameter_jacobian_sparsity(backend)
    c = ExaCore()
    x = variable(c, 3)
    θ = parameter(c, [1.0, 2.0, 3.0])
    
    # Create constraints with specific sparsity pattern
    constraint(c, θ[1] * x[1] + θ[2] * x[2])  # Depends on θ[1], θ[2]
    constraint(c, θ[2] * x[2] + θ[3] * x[3])  # Depends on θ[2], θ[3]
    constraint(c, θ[1] * x[1] + θ[3] * x[3])  # Depends on θ[1], θ[3]
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [1.0, 2.0, 3.0]
    
    # Compute parameter jacobian transpose using jptprod! (J_p^T * e_i)
    param_jac_T = zeros(3, 3)
    for i in 1:3
        adj = zeros(3)
        adj[i] = 1.0
        param_jac_T[i, :] = ExaModels.jptprod!(nlp, x_test, adj, zeros(3))
    end
    
    # Check sparsity pattern
    # param_jac_T[i,:] = J_p^T * e_i, so param_jac_T = J_p^T
    # Expected J_p = [[1,1,0], [0,1,1], [1,0,1]], so J_p^T = [[1,0,1], [1,1,0], [0,1,1]]
    # But we're computing row-wise, so param_jac_T is actually J_p^T transposed = J_p
    expected_pattern = [1 1 0; 0 1 1; 1 0 1]  # Non-zero pattern of J_p  
    actual_pattern = (abs.(param_jac_T) .> 1e-12)
    @test actual_pattern == expected_pattern
end

"""
Test mixed hessian sparsity pattern
"""
function test_mixed_hessian_sparsity(backend)
    c = ExaCore()
    x = variable(c, 3)
    θ = parameter(c, [1.0, 2.0, 3.0])
    
    # Create objective with specific sparsity pattern
    objective(c, θ[1] * x[1]^2 + θ[2] * x[2]^2 + θ[3] * x[3]^2)
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [1.0, 2.0, 3.0]
    
    # Compute mixed hessian (objective only, no constraint multipliers)
    mixed_hess = zeros(3, 3)
    for i in 1:3
        u = zeros(3)
        u[i] = 1.0
        mixed_hess[i, :] = ExaModels.mhtprod!(nlp, x_test, u, zeros(3))
    end
    
    # Check sparsity pattern (should be diagonal for this simple case)
    expected_pattern = [1 0 0; 0 1 0; 0 0 1]
    actual_pattern = (abs.(mixed_hess) .> 1e-12)
    
    @test actual_pattern == expected_pattern
end

"""
Test parameter jacobian with different parameter values
"""
function test_parameter_jacobian_parameter_values(backend)
    c = ExaCore()
    x = variable(c, 2)
    θ = parameter(c, [1.0, 2.0])
    
    constraint(c, θ[1] * x[1] + θ[2] * x[2])
    constraint(c, θ[1] * x[1]^2 + θ[2] * x[2]^2)
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [1.0, 2.0]
    
    # Test with different parameter values
    param_values = [[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]]
    
    for θ_test in param_values
        set_parameter!(c, θ, θ_test)
        
        # Compute parameter jacobian transpose using jptprod! (J_p^T * e_i)
        param_jac_T = zeros(2, 2)
        for i in 1:2
            adj = zeros(2)
            adj[i] = 1.0
            param_jac_T[i, :] = ExaModels.jptprod!(nlp, x_test, adj, zeros(2))
        end
        
        # Expected jacobian:
        # For constraint 1: θ[1] * x[1] + θ[2] * x[2] => ∂c1/∂θ = [x[1], x[2]] = [1, 2]  
        # For constraint 2: θ[1] * x[1]^2 + θ[2] * x[2]^2 => ∂c2/∂θ = [x[1]^2, x[2]^2] = [1, 4]
        # J_p = [[1, 2], [1, 4]], so J_p^T = [[1, 1], [2, 4]]
        # jptprod! returns J_p^T * e_i, so param_jac_T[i,:] = J_p^T * e_i
        # param_jac_T[1,:] = J_p^T * [1,0] = [1, 2]  
        # param_jac_T[2,:] = J_p^T * [0,1] = [1, 4]
        # So param_jac_T = [[1, 2], [1, 4]]
        expected_jac_T = [x_test[1]  x_test[2];
                          x_test[1]^2  x_test[2]^2]
        
        @test param_jac_T ≈ expected_jac_T atol=1e-12
    end
end

"""
Test mixed hessian with different parameter values
"""
function test_mixed_hessian_parameter_values(backend)
    c = ExaCore()
    x = variable(c, 2)
    θ = parameter(c, [1.0, 2.0])
    
    objective(c, θ[1] * x[1]^2 + θ[2] * x[2]^2)
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [1.0, 2.0]
    
    # Test with different parameter values
    param_values = [[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]]
    
    for θ_test in param_values
        set_parameter!(c, θ, θ_test)
        
        # Compute mixed hessian (objective only)
        mixed_hess = zeros(2, 2)
        for i in 1:2
            u = zeros(2)
            u[i] = 1.0
            mixed_hess[i, :] = ExaModels.mhtprod!(nlp, x_test, u, zeros(2))
        end
        
        # Expected mixed hessian:
        # For objective: y[1] * (θ[1] * x[1]^2 + θ[2] * x[2]^2)
        # ∂²obj/∂x[1]∂θ = [2*y[1]*x[1], 0] = [2*1*1, 0] = [2, 0]
        # ∂²obj/∂x[2]∂θ = [0, 2*y[1]*x[2]] = [0, 2*1*2] = [0, 4]
        expected_hess = [2*x_test[1] 0.0; 0.0 2*x_test[2]]
        
        @test mixed_hess ≈ expected_hess atol=1e-12
    end
end

"""
Test mhprod! (mixed Hessian times parameter vector) with a simple example
"""
function test_mhprod_simple(backend)
    c = ExaCore()
    x = variable(c, 2)
    θ = parameter(c, [1.0, 2.0])
    
    # Objective: f = θ[1] * x[1]^2 + θ[2] * x[1] * x[2]
    # Mixed Hessian H = ∂²f/∂x∂θ:
    # ∂f/∂x = [2*θ[1]*x[1] + θ[2]*x[2], θ[2]*x[1]]
    # H = [[2*x[1], x[2]], [0, x[1]]]
    # At x = [1, 2]: H = [[2, 2], [0, 1]]
    objective(c, θ[1] * x[1]^2 + θ[2] * x[1] * x[2])
    
    # Add a simple constraint with parameters to test lagrangian mixed hessian
    constraint(c, θ[1] * x[1] + θ[2] * x[2] - 3.0)
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [1.0, 2.0]
    v_param = [1.0, 0.5]
    y_test = [0.0]  # Zero multiplier to test objective-only case
    
    # Test mhprod: H * v_param (objective only)
    result_mhprod = zeros(2)
    ExaModels.mhprod!(nlp, x_test, y_test, v_param, result_mhprod; obj_weight=1.0)
    
    # Expected: H * [1, 0.5] = [[2, 2], [0, 1]] * [1, 0.5] = [2*1 + 2*0.5, 0*1 + 1*0.5] = [3.0, 0.5]
    expected = [3.0, 0.5]
    
    @test result_mhprod ≈ expected atol=1e-12
    
    # Test with non-zero multiplier to include constraint contribution
    y_test_nonzero = [1.0]
    result_mhprod_constraint = zeros(2)
    ExaModels.mhprod!(nlp, x_test, y_test_nonzero, v_param, result_mhprod_constraint; obj_weight=1.0)
    
    # For constraint θ[1]*x[1] + θ[2]*x[2] - 3.0, the mixed Hessian contribution is:
    # ∂²/∂x∂θ[y * (θ[1]*x[1] + θ[2]*x[2] - 3.0)] = y * [[1, 0], [0, 1]]
    # At y = [1]: contribution = [[1, 0], [0, 1]]
    # Total H = objective + constraint = [[2, 2], [0, 1]] + [[1, 0], [0, 1]] = [[3, 2], [0, 2]]
    # H * [1, 0.5] = [[3, 2], [0, 2]] * [1, 0.5] = [3*1 + 2*0.5, 0*1 + 2*0.5] = [4.0, 1.0]
    expected_with_constraint = [4.0, 1.0]
    
    @test result_mhprod_constraint ≈ expected_with_constraint atol=1e-12
    # Finite-difference check (objective-only and with constraint)
    θval = [1.0, 2.0]
    fd = zeros(2); fd_mhprod!(nlp, c, θ, θval, x_test, zeros(0), v_param, fd; obj_weight=1.0)
    @test result_mhprod ≈ fd atol=1e-4
    fd2 = zeros(2); fd_mhprod!(nlp, c, θ, θval, x_test, y_test_nonzero, v_param, fd2; obj_weight=1.0)
    @test result_mhprod_constraint ≈ fd2 atol=1e-4
end

"""
Test mhprod! consistency with mhtprod! through the relationship (H*v)^T = v^T*H^T
"""
function test_mhprod_consistency(backend)
    c = ExaCore()
    x = variable(c, 2)
    θ = parameter(c, [0.5, 1.5])
    
    # More complex objective with cross terms
    objective(c, θ[1] * x[1]^3 + θ[2] * x[1]^2 * x[2] + θ[1] * θ[2] * x[2]^2)
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [1.5, 2.5]
    v_param = [2.0, 1.0]
    v_var = [1.0, 0.5]
    
    # Test mhprod: H * v_param (parameter space → variable space)
    result_mhprod = zeros(2)
    ExaModels.mhprod!(nlp, x_test, zeros(0), v_param, result_mhprod; obj_weight=1.0)
    
    # Test mhtprod: H' * v_var (variable space → parameter space)
    result_mhtprod = zeros(2)
    ExaModels.mhtprod!(nlp, x_test, zeros(0), v_var, result_mhtprod; obj_weight=1.0)
    
    # Test the relationship: (H * v_param) · v_var = v_param · (H' * v_var)
    dot_mhprod = sum(result_mhprod .* v_var)
    dot_mhtprod = sum(v_param .* result_mhtprod)
    
    @test dot_mhprod ≈ dot_mhtprod atol=1e-12
    
    # Finite-difference checks for both forms
    θval = [0.5, 1.5]
    fd = zeros(2); fd_mhprod!(nlp, c, θ, θval, x_test, zeros(0), v_param, fd; obj_weight=1.0)
    @test result_mhprod ≈ fd atol=1e-4
    fdT = zeros(2); fd_mhtprod!(nlp, c, θ, θval, x_test, zeros(0), v_var, fdT; obj_weight=1.0)
    @test result_mhtprod ≈ fdT atol=1e-4
end

"""
Test mhprod! with constraints (lagrangian mixed Hessian)
"""
function test_mhprod_with_constraints(backend)
    c = ExaCore()
    x = variable(c, 3)
    θ = parameter(c, [2.0, 1.0, 0.5])
    
    # Objective with parameter dependencies
    objective(c, θ[1] * x[1]^2 + θ[2] * x[2]^2 + θ[3] * x[1] * x[2])
    
    # Constraints with parameter dependencies
    constraint(c, θ[1] * x[1] + θ[2] * x[2] + x[3] - 1.0)
    constraint(c, x[1]^2 + θ[3] * x[2]^2 - θ[1])
    
    m = ExaModel(c; prod=true)
    nlp = WrapperNLPModel(m)
    
    x_test = [0.5, 1.0, 0.2]
    v_param = [1.0, 0.5, 2.0]
    v_var = [1.0, 0.5, 0.25]
    y_test = [0.5, 1.5]  # Lagrange multipliers
    
    # Test mhprod with constraints: H * v_param
    result_mhprod = zeros(3)
    ExaModels.mhprod!(nlp, x_test, y_test, v_param, result_mhprod; obj_weight=1.0)
    
    # Test mhtprod with constraints: H' * v_var
    result_mhtprod = zeros(3)
    ExaModels.mhtprod!(nlp, x_test, y_test, v_var, result_mhtprod; obj_weight=1.0)
    
    # Bilinear identity with constraints
    @test abs(sum(result_mhprod .* v_var) - sum(v_param .* result_mhtprod)) ≤ 1e-10
    # Finite-difference checks with constraints
    θval = [2.0, 1.0, 0.5]
    fd = zeros(3); fd_mhprod!(nlp, c, θ, θval, x_test, y_test, v_param, fd; obj_weight=1.0)
    @test result_mhprod ≈ fd atol=1e-4
    fdT = zeros(3); fd_mhtprod!(nlp, c, θ, θval, x_test, y_test, v_var, fdT; obj_weight=1.0)
    @test result_mhtprod ≈ fdT atol=1e-4
    # Test different multipliers give different results
    y_test2 = [1.0, 0.5]
    result_mhprod2 = zeros(3)
    ExaModels.mhprod!(nlp, x_test, y_test2, v_param, result_mhprod2; obj_weight=1.0)
    @test !(result_mhprod ≈ result_mhprod2)
end

"""
Main test function for parameter derivatives
"""
function test_parameter_derivatives(backend)
    @testset "Parameter Jacobian Tests" begin
        
        @testset "Consistency Check" begin
            test_parameter_jacobian_consistency(backend)
        end
        
        @testset "Sparsity Pattern" begin
            test_parameter_jacobian_sparsity(backend)
        end
        
        @testset "Parameter Values" begin
            test_parameter_jacobian_parameter_values(backend)
        end
    end
    
    @testset "Mixed Hessian Tests" begin
        
        @testset "Sparsity Pattern" begin
            test_mixed_hessian_sparsity(backend)
        end
        
        @testset "Parameter Values" begin
            test_mixed_hessian_parameter_values(backend)
        end
        
        @testset "mhprod! (H*v)" begin
            test_mhprod_simple(backend)
            test_mhprod_consistency(backend)
            test_mhprod_with_constraints(backend)
        end
    end
end
