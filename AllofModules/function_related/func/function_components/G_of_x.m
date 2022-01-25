function [H]                           = G_of_x(x_k)
global cnstData
       G          = reshape(x_k.u(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
       H          = G(1:cnstData.nSDP-1,1:cnstData.nSDP-1);
end