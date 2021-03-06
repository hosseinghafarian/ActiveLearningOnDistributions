function [y,v] = saddle_solver(x_0, u_0, sec_order_oracle, project, L_xx, L_uu, max_iter, conv_meas)

iter      = 1;
converged = false;
y         = x_0; 
v         = u_0;
while ~converged && (iter <= max_iter)
    [grad_x, Hessian_xx, grad_u, Hessian_uu, Hessian_xu ] =sec_order_oracle(y,v); 
    M_xx_inv = inv(Hessian_xx);
    M_uu_inv = inv(Hessian_uu);
    A_u      = Hessian_xu'*1/L_xx*Hessian_xu + L_uu*eye(size(v));
    %A_u      = Hessian_xu'*M_xx_inv*Hessian_xu + L_uu*eye(size(v));
    A_x      = -Hessian_xu *1/L_uu*Hessian_xu'+ L_xx*eye(size(y));
    %A_x      = -Hessian_xu *M_uu_inv*Hessian_xu'+ L_xx*eye(size(y));
    a_x      = grad_x-grad_u*M_uu_inv*Hessian_xu;
    a_u      = -(grad_u-grad_x*M_xx_inv*Hessian_xu');
    x_k      = y - A_x\a_x;
    u_k      = v - A_u\a_u; 
    [x_k,u_k]= project(x_k, u_k);
    dnorm    = norm(y-x_k)+norm(v-u_k);
    if dnorm <= conv_meas
        converged = true;
    end
    y        = x_k;
    v        = u_k;
    iter     = iter + 1;
end

end