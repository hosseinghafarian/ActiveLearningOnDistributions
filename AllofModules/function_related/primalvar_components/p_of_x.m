function [p]                           = p_of_x(x_k)
global cnstData
       p          = x_k.u(cnstData.nSDP*cnstData.nSDP+1:end);
end