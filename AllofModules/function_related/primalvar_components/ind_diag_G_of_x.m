function [ind_dG]                     = ind_diag_G_of_x(x_k, nSDP)
global cnstData
if nargin <2 
    nSDP = cnstData.nSDP;
end
   ind        = 1:nSDP;
   ind_dG     = sub2ind([nSDP,nSDP], ind,ind);
end