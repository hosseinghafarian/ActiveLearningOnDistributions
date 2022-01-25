function [g_x, g_alpha, H_alphax_u, H_alphax_w_obeta, H_alphax_st]  = f_xAlpha_semiHessian(x_k,alpha_k,learningparams)
global cnstData
    n_x_st     = numel(x_k.st);
    n_alpha    = numel(alpha_k);
    g_x.u      = M_of_alpha(alpha_k,learningparams);
    g_x.w_obeta= learningparams.lambda_o*cnstData.K*x_k.w_obeta;
    g_x.st     = zeros(n_x_st,1);
    g_alpha    = l_of_x(x_k) -1/(learningparams.lambda)*(cnstData.KE.*G_of_x(x_k))*alpha_k;   
    H_alphax_u       = diag(cl_of_x - 1/(learningparams.lambda)*(cnstData.KE)*alpha_k)*ones(n_alpha,cnstData.nConic);
    H_alphax_w_obeta = zeros(n_alpha, cnstData.n_S);
    H_alphax_st      = zeros(n_x_st , cnstData.n_S);
end