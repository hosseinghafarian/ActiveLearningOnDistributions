function [dist] = relative_dist_to_x_opt(x_k)
global g_x_opt
global g_x_opt_set
global g_x_gscale
    if g_x_opt_set
        x_k.u       = x_k.u       * g_x_gscale;
        x_k.st      = x_k.st      * g_x_gscale;
        x_k.w_obeta = x_k.w_obeta * g_x_gscale; 
        dist = euclidean_dist_of_x(x_k, g_x_opt)/x_norm(g_x_opt);
    else
        dist = realmax;
    end
end