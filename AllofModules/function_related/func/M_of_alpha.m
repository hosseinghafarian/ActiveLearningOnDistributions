function [M] = M_of_alpha(alpha_k,learningparams)
% This function returns gradient of f(x,\alpha) with respect to x. In this
% f(x,\alpha) is the original objective function. 
global cnstData
typ = 1;
if typ == 1
    sub        = cnstData.unlabeled;
    coeff      = [alpha_k(1:cnstData.n_S);zeros(cnstData.nSDP-1-cnstData.n_S,1)]; % this term is l^*(\alpha)
    coeff(sub) = coeff(sub) + learningparams.ca;                                  % the term ca*1 is \eta^*
    C          = diag(coeff) -1/(2*learningparams.lambda)*(alpha_k*alpha_k').*cnstData.KE;
    MG         = reshape([C,zeros(cnstData.nSDP-1,1);zeros(1,cnstData.nSDP)],cnstData.nSDP*cnstData.nSDP,1);
    Mp         = learningparams.cp*ones(cnstData.n_S,1);
    M          = [MG;Mp];
elseif typ == 2
    C          = -1/(2*learningparams.lambda)*(alpha_k*alpha_k').*cnstData.KE; % this term is G^*
    MG         = reshape([C,zeros(cnstData.nSDP-1,1);zeros(1,cnstData.nSDP)],cnstData.nSDP*cnstData.nSDP,1);
    Mp         = zeros(cnstData.n_S,1);
    M          = [MG;Mp];
    M          = l_star(alpha_k) + eta_star(learningparams) + M;
end
end