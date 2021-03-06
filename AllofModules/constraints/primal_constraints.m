function [yal_primal_constraints] = primal_constraints(x_k, scoperator)
global cnstData
[X,p,w_obeta,st,q,qyu]  = parts_of_x(x_k);
s_IC_indices   = 1:scoperator.n_AIC;
s_IV_indices   = scoperator.n_AIC+1:scoperator.n_AIC+scoperator.n_AIV;
yal_primal_constraints= [X>=0,...
                           p>=0,...
                           scoperator.A_EC*x_k.u ==scoperator.b_EC,...
                           scoperator.A_IC*x_k.u ==x_k.st(s_IC_indices),...
                           scoperator.A_IV*x_k.u + scoperator.B_IV*x_k.w_obeta == x_k.st(s_IV_indices),...
                           scoperator.A_EV*x_k.u + scoperator.B_EV*x_k.w_obeta == scoperator.b_EV,...
                           x_k.st(s_IC_indices)<=scoperator.s_IC,...
                           x_k.st(s_IV_indices)<=scoperator.s_IV, ...
                           X(cnstData.extendInd,cnstData.nSDP)>=0, ...
                           X(cnstData.nSDP,cnstData.extendInd)>=0 ];
end