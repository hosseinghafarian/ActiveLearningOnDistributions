function [x_k] = x_conv_u(u,w_obeta,st)
    x_k.u       = u;
    x_k.w_obeta = w_obeta;
    x_k.st      = st;
end