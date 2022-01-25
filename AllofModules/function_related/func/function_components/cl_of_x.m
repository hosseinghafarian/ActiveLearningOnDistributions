function [l]                           = cl_of_x
global cnstData
       h          = ones(cnstData.n_S,1);
       l          = [h;zeros(cnstData.nSDP-1-cnstData.n_S,1)];  
end
