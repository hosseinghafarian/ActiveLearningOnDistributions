function [ lin_approx ] = linear_approximator_eval(x, x_0, acc_f, acc_g_inner_x, acc_grad, A_kplus, f_func, Psi_func, dist_func, operators, rho_x)
    lin_approx     = 1/2*dist_func(x,x_0) ;
    lin_approx     = lin_approx +  acc_grad.u'*x.u + acc_grad.w_obeta'*x.w_obeta + acc_grad.st'*x.st;
    lin_approx     = lin_approx + acc_f - acc_g_inner_x;
    lin_approx     = lin_approx + A_kplus*Psi_func(x, x_0, operators, rho_x);
end