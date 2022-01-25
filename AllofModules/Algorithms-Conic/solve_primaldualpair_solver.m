function [x_k,dualvars_k, comp, feas ] = solve_primaldualpair_solver(objectivefunc,x_0,operators,learningparams,optparams)
%% This function Solves least squares SDP x\in \mathcal{C} problem using a solver and yalmip
%  it doesn't scale/descale problem. If the problem must be scaled, it must
%  be scaled before calling this function and descaled after call returns. 
    %% Setting Proximal parameters 
    tolgap = 10^-4;
    %setting proximal x^k for now, just fot the test
    global cnstData;

    objectivefunc_primal = objectivefunc.primal;
    objectivefunc_dual   = objectivefunc.dual; 
    [x_k,      obj_val_primal, sol_problem_primal] = solve_primal(objectivefunc_primal,x_0, operators, learningparams, optparams);
    [dualvars_k, obj_val_dual, sol_problem_dual]   = solve_dual  (objectivefunc_dual  ,x_0, operators, learningparams, optparams);

    if abs(obj_val_primal-obj_val_dual)/(1+abs(obj_val_primal)) > tolgap, assert(true,'gap is not zero in function solvePsi_k4'),end
    %% Check primal-dual pair
    [x_k_from_dual ]                           = x_conv_from_dual(dualvars_k, x_0, operators);
    if euclidean_dist_of_x(x_k,x_k_from_dual)/x_norm(x_k) > tolgap, assert(true,'primal and dual variables do not match');end
    [comp.yAEC, comp.yAEV,comp.yAIC,comp.yAIV,...
              comp.SDP , comp.V   ,...
              feas.AEC,feas.AIC,feas.AEV,feas.AIV] = checkComplementarity(operators, x_k, dualvars_k);
    [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_grad(dualvars_k, x_0, operators, learningparams, optparams);
end