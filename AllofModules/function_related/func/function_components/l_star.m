function [M]                           = l_star(alpha_k)
global cnstData
       coeff      = [alpha_k(1:cnstData.n_S);zeros(cnstData.nSDP-1-cnstData.n_S,1)]; % this term is l^*(\alpha)
       C          = diag(coeff);
       MG         = reshape([C,zeros(cnstData.nSDP-1,1);zeros(1,cnstData.nSDP)],cnstData.nSDP*cnstData.nSDP,1);
       Mp         = zeros(cnstData.n_S,1);
       M          = [MG;Mp];
end