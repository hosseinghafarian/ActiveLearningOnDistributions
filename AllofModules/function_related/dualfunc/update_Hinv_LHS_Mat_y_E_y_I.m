function [ LHS_y_E, LHS_y_I, CHOL_H_y_E, CHOL_H_y_I ] = update_Hinv_LHS_Mat_y_E_y_I(operators)
global cnstData
    A_E        = [operators.A_EC;operators.A_EV];
    LHS_y_E    = A_E*A_E' + operators.B_E*cnstData.Hinv*operators.B_E';
    CHOL_H_y_E = chol(LHS_y_E);
    LHS_y_I    = operators.AA_I + operators.B_I*cnstData.Hinv*operators.B_I'+eye(size(operators.AA_I));
    LHS_y_I    = (LHS_y_I + LHS_y_I)/2;
    [CHOL_H_y_I] = chol(LHS_y_I);
end