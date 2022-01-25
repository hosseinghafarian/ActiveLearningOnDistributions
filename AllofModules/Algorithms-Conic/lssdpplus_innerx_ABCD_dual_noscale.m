function [ dualvars_next, x_next, solstatus ] = lssdpplus_innerx_ABCD_dual_noscale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
                      x_0, dualvarsPre, g_acc_x, beta_kx, operators,learningparams,optparams)
%% This function solves the problem: min_{x\in \mathcal{C}} <g_acc_x,x> + beta_kx/2* \Vert x-x_0 \Vert^2
%       The set \mathcal{C} is defined by using operators. It calls
%       lssdp_ABCD_dualscaledescale function 
%       max_conv_meas is the maximum amount of etall if it exits
%       optimization loop sooner than max_iter. 
%       It will exit optimization loop sooner, if it feels there is no
%       progress which is determined if there is no enough change in last
%       noprogress_duration iterations. 
%       x_k : is the primal variable returned
%       dualvars_k: is the dual variable returned
%       There is no scaling in this function                  
%%
global cnstData
% form least square: \Vert x-x_prox_0 \Vert^2
x_prox_0.u           = (1/beta_kx)*(- g_acc_x.u                     + beta_kx*x_0.u      );
x_prox_0.w_obeta     = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_0.w_obeta);
x_prox_0.st          = (1/beta_kx)*(- g_acc_x.st                    + beta_kx*x_0.st     );
update_LHS_Mat_y_E_y_I = objectivefunc.update_LHS_Mat_y_E_y_I;
sig_k_coeff = learningparams.lambda_o/learningparams.rhox;
Q_coeff     = 1;
update_inv_H(sig_k_coeff, Q_coeff);
[operators.LHS_y_E, operators.LHS_y_I, operators.CHOL_H_y_E, operators.CHOL_H_y_I] ...
           = update_LHS_Mat_y_E_y_I(operators);
[dualvars_next, x_next, solstatus ]...
     = lssdp_ABCD_dual_noscale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
                      x_prox_0, dualvarsPre, operators,learningparams,optparams);                  
   
end