function [Mat, RHS] = dual_objective_split_LHS_RHS(retType, b_E, y_EC, y_EV, y_IC, y_IV, S, Z , v, z_hat, operators, learningparams, optparams)
global cnstData 

    c_SZ              = S + Z + z_hat.u; % assumed that \mathcal{S}_u = I
    if retType == 1 || retType ==2 % y_E 
        y_I           = [y_IC;y_IV];
        RHS           = b_E - operators.A_E*(operators.A_I'*y_I + c_SZ)...
                            - operators.B_E*(cnstData.Qinv*operators.B_I'*y_I +z_hat.w_obeta);
        if retType ==1             % pcg
            Mat       = operators.LHS_E;
        else% retType ==2          % chol
            Mat       = operators.CHOL_H_y_E;
        end
    elseif retType == 3 || retType == 4 % y_I
        y_E           = [y_EC;y_EV];
        RHS           = z_hat.st-v-operators.A_I*(operators.A_E'*y_E+c_SZ)...
                                  -operators.B_I*(cnstData.Qinv*operators.B_E'*y_E +z_hat.w_obeta);
        if retType == 3            % pcg
            Mat       = operators.LHS_I;
        else% retType == 4         % chol
            Mat       = operators.CHOL_H_y_I;
        end
    end
    %[f_dual] = dual_objective_split(y_EC, y_EV, y_IC, y_IV, S, Z , v, z_hat, operators, learningparams, optparams);
end