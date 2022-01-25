function [ LHS_y_E, LHS_y_I, CHOL_H_y_E, CHOL_H_y_I ] = update_Qinv_LHS_Mat_y_E_y_I(operators)
%% This function computes Left Hand Side and cholesky factorization for computing y_E and y_I, using operators and global cnstData
global cnstData
    A_E          = operators.A_E;
    B_E          = operators.B_E;
    B_I          = operators.B_I;
    AA_I         = operators.AA_I;
    LHS_y_E      = A_E*A_E' + B_E*cnstData.Qinv*B_E';
    CHOL_H_y_E   = chol(LHS_y_E);
    LHS_y_I      = AA_I + B_I*cnstData.Qinv*B_I'+eye(size(AA_I));
    LHS_y_I      = (LHS_y_I + LHS_y_I)/2;
    [CHOL_H_y_I] = chol(LHS_y_I);
end