function [x_next,dualvars_k]=testdualanefficientinexact10(learningparams,operators,dualvars,...
                                                            x_G,alphapre,c_k)
%% this is the complete form of the test for primal dual problem, with both w_o and Z, it works fine.  

%% Setting Proximal parameters 
%setting proximal x^k for now, just fot the test
global cnstData;

    u_t       = x_G.u;
    s_t       = x_G.st;
    w_obetat  = x_G.w_obeta;
    Gprox     = u_t+c_k/arho;
    gprox     = s_t;
    gammanorm = max(norm(Gprox),norm(gprox));
    Gprox     = Gprox /gammanorm;
    gprox     = gprox /gammanorm;
    w_obetat  = w_obetat/gammanorm;
    Ghat. u      = ?
    Ghat.w_obeta = ?
    Ghat.st      = ?
    %% Primal problem 
    [gscale,scoperators]                      = scaleProblem(learningparams,operators,Ghat);
    [x_k, obj_val_primal, sol_problem_primal] = solve_primal_leastSquares(Ghat,scoperators,learningparams,optparams);
    [x_next] = descale_in_primal(gscale, learningparams, operators,x_k);

    %% Dual problem 
    [dualvars_k, obj_val_dual, sol_problem_dual] = solve_dual_leastSquares_Constraints(Ghat,scoperators,learningparams,optparams);
    [dualvars_k ] = descale_in_dual(gscale, learningparams, operators, dualvars_k);

    if abs(obj_val_primal-obj_val_dual)/(1+abs(obj_val_primal)) > tolgap, assert(true,'gap is not zero in function solvePsi_k4'),end
    %% Check primal-dual pair
    [x_k_from_dual ]                           = x_conv_from_dual(dualvars_k, Ghat, scoperators);
    [comp.yAEC, comp.yAEV,comp.yAIC,comp.yAIV,...
              comp.SDP , comp.V   ,...
              feas.AEC,feas.AIC,feas.AEV,feas.AIV] = checkComplementarity(scoperators, x_k, dualvars_k);
    [eqIneq.equalitydiff, eqIneq.eq, eqIneq.Inequality, eqIneq.Ineq]       = ConstraintsCheck(x_k, learningparams);
end
