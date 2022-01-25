function [g_x, g_alpha, H_alphax_u, H_alphax_w_obeta, H_alphax_st]  = f_xAlpha_regularized_semiHessian(x_k,alpha_k,learningparams, x_0, alpha_0)
global cnstData
    n_x_st     = numel(x_k.st);
    n_alpha    = numel(alpha_k);
    g_x.u      = M_of_alpha(alpha_k,learningparams)              + learningparams.rhox*u_grad_euclidean_dist_of_x;
    g_x.w_obeta= learningparams.lambda_o*cnstData.K*x_k.w_obeta  + learningparams.rhox*wo_grad_euclidean_dist_of_x;
    g_x.st     =                                                   learningparams.rhox*st_grad_euclidean_dist_of_x;
    g_alpha    = l_of_x(x_k) -1/(learningparams.lambda)*(cnstData.KE.*G_of_x(x_k))*alpha_k + learningparams.rhoalpha*(alpha_k-alpha_0);   
    H_alphax_u       = diag(cl_of_x - 1/(learningparams.lambda)*(cnstData.KE)*alpha_k)*ones(n_alpha,cnstData.nConic);
    H_alphax_w_obeta = zeros(n_alpha, cnstData.n_S);
    H_alphax_st      = zeros(n_alpha, n_x_st);
    function [gu ]   = u_grad_euclidean_dist_of_x
       gu   =  x_k.u-x_0.u;
    end
    function [gwo ]   = wo_grad_euclidean_dist_of_x
       gwo   = cnstData.Q*(x_k.w_obeta-x_0.w_obeta);
    end
    function [gst ]   = st_grad_euclidean_dist_of_x
       gst   = (x_k.st-x_0.st);
    end
end