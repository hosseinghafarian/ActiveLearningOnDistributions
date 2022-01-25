function [q]                           = q_of_x(x_k)
global cnstData
       ind       = ind_of_q_in_x(); 
       q_smal    = x_k.u(ind);
       q         = zeros(cnstData.n_S,1);
       q(cnstData.query)  = q_smal;
end