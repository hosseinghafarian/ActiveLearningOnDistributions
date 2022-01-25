function [acc_f, acc_g_inner_x, g_acc_x, normg] = linear_approximator_update...
                     (x_k, a_k, acc_f, acc_g_inner_x, g_acc_x, f_func, Psi_func, dist_func, operators)
    [ f_xkplus, g_f ] = f_func(x_k, operators);
    normg             = x_norm(g_f);
    acc_f             = acc_f           + a_k*f_xkplus;
    acc_g_inner_x     = acc_g_inner_x   + a_k*x_inner(g_f,x_k);
    g_acc_x.u         = g_acc_x.u       + a_k* g_f.u;
    g_acc_x.w_obeta   = g_acc_x.w_obeta + a_k* g_f.w_obeta;
    g_acc_x.st        = g_acc_x.st      + a_k* g_f.st;
end