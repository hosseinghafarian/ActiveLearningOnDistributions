function [ lin_approx, acc_f, acc_g_inner_x, acc_grad] = linear_approximator(x, x_kplus, A_kplus, a_kplus, acc_f, acc_g_x, acc_grad, f_func, Psi_func, dist_func, operators)
    [ f_xkplus, g_f_xkplus ] = f_func(x_kplus, operators);
    acc_f          = acc_f   + a_kplus*f_xkplus;
    acc_g_inner_x  = acc_g_x + a_kplus*x_inner(g_f_xplus,x_kplus);
    acc_grad       = x_sum(acc_grad, x_mul(g_f_xkplus,a_kplus));
    lin_approx     = 1/2*dist_func(x,x_0) + ...
                     acc_f +  x_inner(acc_grad,x)- acc_g_inner_x + A_kplus*Psi_func(x_kplus, x_0, operators, rho_x);
end