function [ L_y_E, L_y_I, L_S, L_Z, L_v] = dual_objective_lipschitz(operators) 
    L_y_E = max(eigs(operators.A_E*operators.A_E',1),eigs(operators.B_E*operators.B_E',1));
    L_y_I = max(eigs(operators.A_I*operators.A_I',1),eigs(operators.B_I*operators.B_I',1));
    L_S   = 1;
    L_Z   = 1;
    L_v   = 1;
end