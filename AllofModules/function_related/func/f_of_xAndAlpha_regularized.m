function [f]                           = f_of_xAndAlpha_regularized(x_k,alpha_k,learningparams,type,x_0,alpha_0)
global cnstData
   f = f_of_xAndAlpha(x_k,alpha_k,learningparams,type) - learningparams.rhoalpha/2*norm(alpha_k-alpha_0)^2 ...
       + learningparams.rhox/2*euclidean_dist_of_x(x_k, x_0);
end