function [ dualvars_next, x_next, solstatus ] = lssdpplus_innerx_ABCD_dual_scaledescale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
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
                  
[dualvars_next, x_next, solstatus ]...
     = lssdp_ABCD_dual_scaledescale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
                      x_prox_0, dualvarsPre, operators,learningparams,optparams);                  
   
end