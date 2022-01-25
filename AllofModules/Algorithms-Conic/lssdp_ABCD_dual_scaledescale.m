function  [dualvars_k, x_k, solstatus] ...
                = lssdp_ABCD_dual_scaledescale (max_conv_meas, max_rel_gap, max_iter, objectivefunc, x_0, dualvars_pre, operators,learningparams,optparams)          

 
%% This function solves the problem: min_{x\in \mathcal{C}} 1/2* \Vert x-x_0 \Vert^2
%       The set \mathcal{C} is defined by using operators. 
%       max_conv_meas is the maximum amount of etall if it exits
%       optimization loop sooner than max_iter. 
%       It will exit optimization loop sooner, if it feels there is no
%       progress which is determined if there is no enough change in last
%       noprogress_duration iterations. 
%       x_k : is the primal variable returned
%       dualvars_k: is the dual variable returned
%       This function scale/descale problem in order to better handle numerical problems
%% scale operators
[scx_0, scdualvars_pre, gscale,scoperators]                      = scaleProblem(learningparams, operators, x_0, dualvars_pre);
update_LHS_Mat_y_E_y_I = objectivefunc.update_LHS_Mat_y_E_y_I;
% setup Hinv and LHS RHS for y_E and y_I solvers
sig_k_coeff = learningparams.lambda_o/learningparams.rhox;
Q_coeff     = 1;
update_inv_H(sig_k_coeff, Q_coeff);
[scoperators.LHS_y_E, scoperators.LHS_y_I, scoperators.CHOL_H_y_E, scoperators.CHOL_H_y_I] ...
           = update_LHS_Mat_y_E_y_I(scoperators);
% call lssdp solver
[dualvars_k, x_k, solstatus]...
           = lssdp_ABCD_dual_noscale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
                                     scx_0, scdualvars_pre, scoperators,learningparams,optparams);
% descale both primal and dual variables 
[x_k]         = descale_in_primal(gscale, learningparams, scoperators  , x_k);
[dualvars_k ] = descale_in_dual  (gscale, learningparams, scoperators, dualvars_k);
end