function [psi_x , subgrad_psi_x] = Psi_func_xconic_distx_x_0(x_k, x_0, operators, rho_x)
   s_I = [operators.s_IC;operators.s_IV];
%    if ~isincones(x_k, s_I)
%        psi_x         = realmax;
%        subgrad_psi_x = realmax;
%        return
%    end
   psi_x = rho_x/2*euclidean_dist_of_x(x_k, x_0);
   subgrad_psi_x.u       = rho_x*(x_k.u       - x_0.u);
   subgrad_psi_x.w_obeta = rho_x*(x_k.w_obeta - x_0.w_obeta);
   subgrad_psi_x.st      = rho_x*(x_k.st      - x_0.st);
end