function [f,g_x,g_alpha]  = f_xAlpha_grad(x_k,alpha_k,learningparams)
global cnstData
    g_x.u      = M_of_alpha(alpha_k,learningparams);
    g_x.w_obeta= learningparams.lambda_o*cnstData.K_o*x_k.w_obeta;
    g_x.st     = zeros(numel(x_k.st),1);
    g_alpha    = l_of_x(x_k) -1/(learningparams.lambda)*(cnstData.KE.*G_of_x(x_k))*alpha_k;            
    % this is a linear approximator of function f 
%     fL         = g_x.u'*x_k.u + 1/2*g_x.w_obeta'*x_k.w_obeta-learningparams.ca*cnstData.n_u;
    f          = f_of_xAndAlpha(x_k,alpha_k,learningparams,3);
end