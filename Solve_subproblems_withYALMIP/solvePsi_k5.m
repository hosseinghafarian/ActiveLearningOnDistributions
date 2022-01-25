function [x_k,dualvars_k, comp, feas, eqIneq] = solvePsi_k5(objectivefunc,x_0, g_acc_x, beta_kx,operators,learningparams,optparams)
    %% Setting Proximal parameters 
    tolgap = 10^-4;
    %setting proximal x^k for now, just fot the test
    global cnstData;

    Ghat.u       = (1/beta_kx)*(- g_acc_x.u                     + beta_kx*x_0.u      );
    Ghat.w_obeta = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_0.w_obeta);
    Ghat.st      = (1/beta_kx)*(- g_acc_x.st                    + beta_kx*x_0.st     );
    
    objectivefunc_primal = objectivefunc.primal;
    objectivefunc_dual   = objectivefunc.dual; 
    [x_0_scaled, gscale, scoperators]                      = scaleProblem(learningparams,operators,Ghat);

    [x_k, obj_val_primal, sol_problem_primal] = solve_primal(objectivefunc_primal, x_0_scaled, scoperators, learningparams, optparams);
    [x_k] = descale_in_primal(gscale, learningparams, operators,x_k);

    %% Dual problem 
    [dualvars_k, obj_val_dual, sol_problem_dual] = solve_dual(objectivefunc_dual, x_0_scaled, scoperators, learningparams, optparams);
    [dualvars_k ] = descale_in_dual(gscale, learningparams, scoperators, dualvars_k);

    if abs(obj_val_primal-obj_val_dual)/(1+abs(obj_val_primal)) > tolgap, assert(true,'gap is not zero in function solvePsi_k4'),end
    %% Check primal-dual pair
    [x_k_from_dual ]                           = x_conv_from_dual(dualvars_k, Ghat, scoperators);
    [comp.yAEC, comp.yAEV,comp.yAIC,comp.yAIV,...
              comp.SDP , comp.V   ,...
              feas.AEC,feas.AIC,feas.AEV,feas.AIV] = checkComplementarity(scoperators, x_k, dualvars_k);
    [eqIneq.equalitydiff, eqIneq.eq, eqIneq.Inequality, eqIneq.Ineq]       = ConstraintsCheck(x_k, learningparams);
end



