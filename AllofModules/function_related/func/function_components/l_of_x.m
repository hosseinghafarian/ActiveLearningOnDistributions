function [l]                           = l_of_x(x_k, n, nSDP)
global cnstData
if nargin< 2
    n    = cnstData.n_S;
    nSDP = cnstData.nSDP;
end
       dG         = diag_G_of_x(x_k, nSDP);
       h          = dG(1:n); %previously: h=1-p_of_x(x_k)-q_of_x(x_k); and this is wrong, since it has no effect on v in abs(v)+p+q = 1
       l          = [h;zeros(nSDP-1-n,1)];  
end
