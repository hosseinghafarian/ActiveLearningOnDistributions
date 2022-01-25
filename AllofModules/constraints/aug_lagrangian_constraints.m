function [ f_x, grad_x, res_y] = aug_lagrangian_constraints(y_k, x_k, operators, gamma_x )

f_type      = 1; 
operators.A = [operators.A_EC;operators.A_EV;operators.A_IC;operators.A_IV];
operators.B = [operators.B_E;operators.B_I];
% y is [y_EC;y_EV;y_IC;y_IV]
if f_type == 1    
    b_E    = [operators.b_EC;operators.b_EV];
    ind    = numel(b_E)+1:numel(b_E)+numel(x_k.st);
    res_y  = operators.A*x_k.u + operators.B*x_k.w_obeta-[b_E;x_k.st];
    f_x    = gamma_x/2* norm(res_y-y_k)^2;
    grad_x.u       = gamma_x*operators.A'*(res_y-y_k);
    grad_x.w_obeta = gamma_x*operators.B'*(res_y-y_k);
    grad_x.st      = gamma_x*(ones(numel(ind),1)-y_k(ind)); 
end
%rho_EC = 10; rho_EV = 10; rho_IC = 10; rho_IV = 10;    
% elseif type == 2
%     AB     = [operators.A;operators.B;zeros(numel(x_k.st),1)];
%     res_y  = AB*[x_k.u;x_k.w_obeta;x_k.st]-[b_E;x_k.st];
%     f_x    = gamma_x*y'*res_y + gamma_x/2* norm(res_y)^2;
%     grad_x = gamma_x*AB'*y    + gamma_x* ?????????

% else % if f_type == 2 
%     IC     = 1:operators.n_AIC;
%     IV     = operators.n_AIC+1:operators.n_AIC+operators.n_AIV;
%     f_ls_x =  rho_EC/2* norm(operators.A_EC*x_k.u-b_EC)^2 ...
%             + rho_EV/2* norm(operators.A_EV*x_k.u + operators.B_EV*x_k.w_obeta-b_EV)^2 ... 
%             + rho_IC/2* norm(operators.A_IC*x_k.u-x_k.st(IC))^2 ...
%             + rho_IV/2* norm(operators.A_IV*x_k.u + operators.B_IV*x_k.w_obeta-x_k.st(IV))^2;
%     grad_x.u       =  rho_EC* operators.A_EC'*(operators.A_EC*x_k.u-b_EC) ...
%                         + rho_EV* operators.A_EV'*(operators.A_EV*x_k.u + operators.B_EV*x_k.w_obeta-b_EV) ... 
%                         + rho_IC* operators.A_IC'*(operators.A_IC*x_k.u-x_k.st(IC)) ...
%                         + rho_IV* operators.A_IV'*(operators.A_IV*x_k.u + operators.B_IV*x_k.w_obeta-x_k.st(IV));
%     grad_x.w_obeta =   rho_EV* operators.B_EV'*(operators.A_EV*x_k.u + operators.B_EV*x_k.w_obeta-b_EV) ... 
%                         + rho_IV* operators.B_IV'*(operators.A_IV*x_k.u + operators.B_IV*x_k.w_obeta-x_k.st(IV));
%     grad_x.st      = [ rho_IC*(x_k.st(IC)-operators.A_IC*x_k.u);
%                           rho_IV*(x_k.st(IV)-operators.A_IV*x_k.u - operators.B_IV*x_k.w_obeta)]; 

end