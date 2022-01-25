function [f_dual] = dual_objective_split(y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams, norm_x_0)
% This function computes objective function of the dual. it is exactly the
% same as dual_regwo_objective_* with change of H to Q and addition of
% x_0.w_obeta'*K*x_0.w_obeta
global cnstData 

    Aysdp         = operators.A'*[y_EC;y_EV;y_IC;y_IV];
    Bysdp         = operators.B_V'*[y_EV;y_IV];
    y_I           = [y_IC;y_IV];
    s_I           = [operators.s_IC; operators.s_IV];
    f_dual1       = ( operators.b_EC'*y_EC+ operators.b_EV'*y_EV);
    f_dual2       = - 1/2*norm(Aysdp + S + Z + x_0.u)^2;
    f_dual3       = - learningparams.rhox/2*(Bysdp+cnstData.Q*x_0.w_obeta)'*cnstData.Qinv*(Bysdp+cnstData.Q*x_0.w_obeta);
    f_dual4       = - 1/2*norm(v + y_I- x_0.st)^2-v'*s_I;
    if nargin==12
       f_dual5 = 1/2*norm_x_0;
    else
       f_dual5       = 1/2*x_norm(x_0,cnstData.Q) ;
    end
    f_dual        = f_dual1 + f_dual2 + f_dual3 + f_dual4 + f_dual5;
end