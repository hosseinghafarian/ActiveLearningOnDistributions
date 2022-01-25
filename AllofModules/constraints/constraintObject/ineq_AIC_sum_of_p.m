function [R, s, size_of_s]                              = ineq_AIC_sum_of_p()
    % cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
    % Constraint: 1^T p <=n_o
    ap        = ones(cnstData.n_S,1);
    R         = [zeros(cnstData.nSDP*cnstData.nSDP,1)',ap'];
    s         = c_mul_pAndw_o*cnstData.n_o;
    % cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
    % Constraint: 1^T p_l <=n_lbn
    assert(cnstData.n_lbn<=cnstData.n_o,'n_lbn, number of noisy labeled points must be less than or equal to n_o, number of noisy labeled points');
    assert(cnstData.n_lbn>0   ,'n_lbn, is zero and so constraint qualifications are not correct and dual problem is not equivalent to primal');
    size_of_s = 1;
end