function [ x_k, dualvars_next, f_alpha,gradf_alpha,perfProfile,solstatus, conv_meas, rel_gap] ...
    = check_lssdpsolver%(max_conv_meas, max_rel_gap, max_iter,objectivefunc,learningparams, optparams, dualvarsPre,x_G,alpha_k)   
%% Based on algorithm ABCD-1 :paper : An efficient inexact ABCD method for least squares semidefinite programming    
%% global variables
global rhoml
global operators
global cnstData
check_with_solver = true;
saved             = true;
% save('parametersofproxf','max_conv_meas', 'max_rel_gap', 'max_iter','objectivefunc','learningparams', 'optparams', 'dualvarsPre','x_G','alpha_k');
% save('globalvars','operators','cnstData');
load('globalvars','operators','cnstData');
load('parametersofproxf','max_conv_meas', 'max_rel_gap', 'max_iter','objectivefunc','learningparams', 'optparams', 'dualvarsPre','x_G','alpha_k');
objectivefunc.primal      = @primal_regwo_objective;
objectivefunc.dual        = @  dual_regwo_objective;
objectivefunc.dualsplit   = @  dual_regwo_objective_split;
objectivefunc.LHSRHS      = @  dual_regwo_objective_split_LHS_RHS;
objectivefunc.update_LHS_Mat_y_E_y_I = @ update_Hinv_LHS_Mat_y_E_y_I;
objectivefunc.dist_x_opt  = @  relative_dist_to_x_opt;
objectivefunc.ProofChecker = @NestCompProofChecker; 
global g_x_opt_set
global g_x_opt
global g_x_gscale
g_x_opt_set = false;

[c_k]                = M_of_alpha(alpha_k,learningparams);
g_acc_x.u            = c_k;
g_acc_x.w_obeta      = zeros(cnstData.n_S,1);
g_acc_x.st           = 0;
beta_kx              = learningparams.rhox;
x_prox_0.u           = (1/beta_kx)*(- g_acc_x.u                     + beta_kx*x_G.u      );
x_prox_0.w_obeta     = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_G.w_obeta);
x_prox_0.st          = (1/beta_kx)*(- g_acc_x.st                    + beta_kx*x_G.st     );
if check_with_solver && ~saved
    [scx_0, dualvars_pre, gscale,scoperators]                      = scaleProblem(learningparams, operators, x_prox_0,dualvarsPre);
    [x_next_solver,dualvars_next_solver, comp, feas] = solve_primaldualpair_solver(objectivefunc, scx_0, scoperators, learningparams, optparams);
    [x_next_solver]         = descale_in_primal(gscale, learningparams, scoperators, x_next_solver);
    [dualvars_next_solver ] = descale_in_dual  (gscale, learningparams, scoperators, dualvars_next_solver);
    [eqIneq.equalitydiff, eqIneq.eq, eqIneq.Inequality, eqIneq.Ineq]       = ConstraintsCheck(x_next_solver, learningparams);
    g_x_opt     = x_next_solver;
    g_x_opt_set = true;
    g_x_gscale  = 1;
    [f_alpha, g_x, gradf_alpha]           = f_xAlpha_grad(x_next_solver,alpha_k,learningparams);
    etallstart  = 0; etallend = 0;
    perfProfile = 0;
    save('save_x_next_solver','x_next_solver','dualvars_next_solver','g_x_opt','g_x_opt_set');
elseif check_with_solver
    load('save_x_next_solver','x_next_solver','dualvars_next_solver','g_x_opt','g_x_opt_set');
end
operators.domain_of_x_min_x_u = [-ones(cnstData.nSDP*cnstData.nSDP,1);zeros(cnstData.n_S,1)];
operators.domain_of_x_max_x_u = [ ones(cnstData.nSDP*cnstData.nSDP,1);ones(cnstData.n_S,1) ];
[ f_ls_x, grad_ls_x] = f_ls_substitute_of_constraints(x_next_solver, operators );
[scx_0, dualvars_pre, gscale,scoperators]                      = scaleProblem(learningparams, operators, x_prox_0,dualvarsPre);
[dualvars_k, x_k, solstatus] = lssdp_NesterovComp_primal_noscale(max_conv_meas, max_rel_gap, max_iter, objectivefunc,scx_0, dualvars_pre, scoperators,learningparams,optparams);
% % [ dualvars_k, x_k, solstatus ] = lssdpplus_innerx_ABCD_dual_scaledescale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
% %                       x_G, dualvarsPre, g_acc_x, beta_kx, operators,learningparams,optparams);
% [dualvars_next, x_k, solstatus ]...
%      = lssdp_ABCD_dual_scaledescale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
%                       x_prox_0, dualvarsPre, operators,learningparams,optparams);
if check_with_solver 
    reldiff = euclidean_dist_of_x(x_k, x_next_solver)/x_norm(x_next_solver);
    if reldiff > max_conv_meas
        assert(true,'error in computing x_next in routine solve_dual_alg checking made in proxf_X_directADMM');
    end
end
[f_alpha, g_x, gradf_alpha]           = f_xAlpha_grad(x_k,alpha_k,learningparams);
perfProfile = 0;
end