function [checked, Ineq ] = NestCompProofChecker(f_func, Psi_func, dist_func, x_0, x_k, v_k, iter, A_k, a_kplus, acc_f, acc_g_inner_x, acc_grad, operators, rho_x)
global g_x_opt_set
global g_x_opt;
if ~g_x_opt_set
    checked = false;
    return 
end
checked = true;
[phix_k  , grad_f_x, subgrad_psi_x] = phi_func(x_k    , x_0, f_func, Psi_func, operators, rho_x); 
[phix_opt, grad_O_x, subgrad_psi_O] = phi_func(g_x_opt, x_0, f_func, Psi_func, operators, rho_x);
dist_x_opt_x_0 = dist_func(g_x_opt, x_0); 
Ineq.ineq_4_5  = phix_k-phix_opt - dist_x_opt_x_0/(2*A_k);
Ineq.ineq_4_6_1= dist_func(g_x_opt, v_k)-dist_func(g_x_opt, x_0);
Ineq.ineq_4_6_2= dist_func(x_0, v_k)  -2*dist_func(g_x_opt, x_0);
[ lin_approx_at_x_k ] = linear_approximator_eval(x_k, x_0, acc_f, acc_g_inner_x, acc_grad, ...
                                          A_k, f_func, Psi_func, dist_func, operators, rho_x);
[ lin_approx_at_v_k ] = linear_approximator_eval(v_k, x_0, acc_f, acc_g_inner_x, acc_grad, ...
                                          A_k, f_func, Psi_func, dist_func, operators, rho_x);
Ineq.ineq_4_4_1 = A_k*phix_k-lin_approx_at_v_k;
Ineq.ineq_4_4_2_at_x_k = lin_approx_at_x_k - A_k*phix_k - 1/2*dist_func(x_k,x_0);
end