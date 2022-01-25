function [pv] = get_p_of_x(x_k)
pv    = x_k.u(ind_of_p_in_x());
end