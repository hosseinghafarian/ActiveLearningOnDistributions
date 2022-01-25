function [ f_ls_x, grad_ls_x] = f_exp_substitute_of_constraints(x_k, operators )
%% This function is least square substitute for constraints. 
%       It has two types in the first form rho = rho_EC and 
%           f(x)= rho_c exp(\rho_gamma/2 \Vert
%           A*x.u+B*x.w_obeta-[b_E;x.st]\Vert^2)
%                        in the second form 
%           f(x)= \rho_EC/2 \Vert A_EC*x.u-b_EC\Vert^2 + \rho_EV/2 \Vert A_EV*x.u+B_EV*x.w_obeta-b_EV\Vert^2
%                 + \rho_IC/2 \Vert A_IC*x.u-x.st(IC)\Vert^2 + \rho_IV/2 \Vert A_IV*x.u+B_IV*x.w_obeta-x.st(IV)\Vert^2
f_type      = 1; rho_c = 1; rho_gamma = 1/2;
operators.A = [operators.A_EC;operators.A_EV;operators.A_IC;operators.A_IV];
operators.B = [operators.B_E;operators.B_I];
if f_type == 1
    b_E    = [operators.b_EC;operators.b_EV];
    f_ls_x = rho_c*exp(rho_gamma/2* norm(operators.A*x_k.u + operators.B*x_k.w_obeta-[b_E;x_k.st])^2);
    grad_ls_x.u       = rho_c*f_ls_x*operators.A'*(operators.A*x_k.u + operators.B*x_k.w_obeta-[b_E;x_k.st]);
    grad_ls_x.w_obeta = rho_c*f_ls_x*operators.B'*(operators.A*x_k.u + operators.B*x_k.w_obeta-[b_E;x_k.st]);
    grad_ls_x.st      = rho_c*f_ls_x*(x_k.st-operators.A_I*x_k.u - operators.B_I*x_k.w_obeta); 
else % if f_type == 2 
    IC     = 1:operators.n_AIC;
    IV     = operators.n_AIC+1:operators.n_AIC+operators.n_AIV;
    f_ls_x =  rho_EC/2* norm(operators.A_EC*x_k.u-b_EC)^2 ...
            + rho_EV/2* norm(operators.A_EV*x_k.u + operators.B_EV*x_k.w_obeta-b_EV)^2 ... 
            + rho_IC/2* norm(operators.A_IC*x_k.u-x_k.st(IC))^2 ...
            + rho_IV/2* norm(operators.A_IV*x_k.u + operators.B_IV*x_k.w_obeta-x_k.st(IV))^2;
    grad_ls_x.u       =  rho_EC* operators.A_EC'*(operators.A_EC*x_k.u-b_EC) ...
                        + rho_EV* operators.A_EV'*(operators.A_EV*x_k.u + operators.B_EV*x_k.w_obeta-b_EV) ... 
                        + rho_IC* operators.A_IC'*(operators.A_IC*x_k.u-x_k.st(IC)) ...
                        + rho_IV* operators.A_IV'*(operators.A_IV*x_k.u + operators.B_IV*x_k.w_obeta-x_k.st(IV));
    grad_ls_x.w_obeta =   rho_EV* operators.B_EV'*(operators.A_EV*x_k.u + operators.B_EV*x_k.w_obeta-b_EV) ... 
                        + rho_IV* operators.B_IV'*(operators.A_IV*x_k.u + operators.B_IV*x_k.w_obeta-x_k.st(IV));
    grad_ls_x.st      = [ rho_IC*(x_k.st(IC)-operators.A_IC*x_k.u);
                          rho_IV*(x_k.st(IV)-operators.A_IV*x_k.u - operators.B_IV*x_k.w_obeta)]; 
end

end