function [dG]                          = diag_G_of_x(x_k, nSDP, sub, ca)
% Returns diag of matrix part of x_k.u
% with one parameter, just all of diagonal elements
% with two parameter, returns just sub indices of diagonal elements
% with three parameters, returns all of diagonal elements plus ca times of
% 'sub' indices for diagonal elements
global cnstData
if nargin == 1 
    nSDP = cnstData.nSDP;
end
    idG        = ind_diag_G_of_x(x_k, nSDP);
    if nargin <= 2
           dG         = x_k.u(idG);
    elseif nargin == 3
           idG2       = idG(sub);
           dG         = x_k.u(idG2);
    elseif nargin == 4
           dG         = x_k.u(idG);
           dG(sub)    = dG(sub)*(1+ca);
    end
end