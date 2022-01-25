function [M] = eta_star(learningparams)
% This function returns gradient of f(x,\alpha) with respect to x. In this
% f(x,\alpha) is the original objective function. 
global cnstData

    sub        = cnstData.unlabeled;
    coeff      = zeros(cnstData.nSDP-1,1); % this term is l^*(\alpha)
    coeff(sub) = coeff(sub) + learningparams.ca;                                  % the term ca*1 is \eta^*
    C          = diag(coeff);
    MG         = reshape([C,zeros(cnstData.nSDP-1,1);zeros(1,cnstData.nSDP)],cnstData.nSDP*cnstData.nSDP,1);
    Mp         = learningparams.cp*ones(cnstData.n_S,1);
    M          = [MG;Mp];
end