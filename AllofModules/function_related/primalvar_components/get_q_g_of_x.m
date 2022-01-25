function [qv, qyu] = get_q_g_of_x(x_k)
qv   = x_k.u(ind_of_q_in_x());
qyu  = x_k.u(ind_of_g_of_labunlab_in_x());
end