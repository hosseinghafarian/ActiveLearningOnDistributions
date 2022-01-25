function [etax]                        = eta_of_x(x_k,learningparams, unlabeled, n, nSDP)
% attention: this function /operator, no longer contain quadratic term
% about x.w_obeta;
global cnstData
if nargin <5
    unlabeled = cnstData.unlabeled;
    n         = cnstData.n_S;
    nSDP      = cnstData.nSDP;
end
       sub       = unlabeled;
       %h        = ;%previously: 1-p_of_x(x_k)-q_of_x(x_k);
       etax      = learningparams.ca*sum(diag_G_of_x(x_k,nSDP, sub));
end