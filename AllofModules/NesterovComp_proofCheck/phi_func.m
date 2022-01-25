function [phi_val, grad_f_x, subgrad_psi_x ] = phi_func(x_k, x_0, f_func, Psi_func, operators, rho_x)
     [f_ls_x, grad_f_x     ] = f_func(x_k, operators);
     [psi_x , subgrad_psi_x] = Psi_func(x_k, x_0, operators, rho_x);
     phi_val = f_ls_x + psi_x;
end