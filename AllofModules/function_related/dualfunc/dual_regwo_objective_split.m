function [f_dual] = dual_regwo_objective_split(y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams)
global cnstData 

    Aysdp         = operators.A'*[y_EC;y_EV;y_IC;y_IV];
    Bysdp         = operators.B_V'*[y_EV;y_IV];
    y_I           = [y_IC;y_IV];
    s_I           = [operators.s_IC; operators.s_IV];
    f_dual        = ( operators.b_EC'*y_EC+ operators.b_EV'*y_EV)...
                     - 1/2*norm(Aysdp + S + Z + x_0.u)^2 ...
                     - learningparams.rhox/2*(Bysdp+cnstData.H*x_0.w_obeta)'*cnstData.Hinv*(Bysdp+cnstData.H*x_0.w_obeta)...
                     - 1/2*norm(v + y_I- x_0.st)^2-v'*s_I...
                     + 1/2*x_norm(x_0,cnstData.H) + learningparams.lambda_o/(2*learningparams.rhox)*x_0.w_obeta'*cnstData.K*x_0.w_obeta;
end