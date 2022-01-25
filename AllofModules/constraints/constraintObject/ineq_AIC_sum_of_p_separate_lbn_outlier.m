function [R, s, size_of_s]                             = ineq_AIC_sum_of_p_separate_lbn_outlier(c_mul_pAndw_o)
    % cConstraint=[cConstraint,sum(p(all-initL)<=n_u*onoiseper/100]; %sum(p(initL))<=n_l*lnoiseper/100,
    % Constraint: 1^T p(initL) <= n_l*lnoiseper/100, 1^T p(all-initL) <= n_u*onoiseper/100
global cnstData
    ap        = zeros(cnstData.n_S,1);
    initL     = cnstData.initL(cnstData.initL>0);
    ap(initL) = 1;
    R(:,1)    = [zeros(1,cnstData.nSDP*cnstData.nSDP),ap'];
    s(1)      = c_mul_pAndw_o*(cnstData.n_l*cnstData.lnoiseper)/100;

    ap        = ones(cnstData.n_S,1);
    ap(initL) = 0; % all ones except for labeled instances which they are in label noise part in the above three lines
    R(:,2)    = [zeros(1,cnstData.nSDP*cnstData.nSDP),ap'];
    s(2)      = c_mul_pAndw_o*(cnstData.n_u*cnstData.onoiseper)/100;
    % cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
    % Constraint: 1^T p_l <=n_lbn
%         assert(cnstData.n_lbn<=cnstData.n_o,'n_lbn, number of noisy labeled points must be less than or equal to n_o, number of noisy labeled points');
%         assert(cnstData.n_lbn>0   ,'n_lbn, is zero and so constraint qualifications are not correct and dual problem is not equivalent to primal');
    size_of_s = 2;
end