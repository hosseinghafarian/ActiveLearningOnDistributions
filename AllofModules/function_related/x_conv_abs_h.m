function [x_k] = x_conv_abs_h(G,p, g_D_neg, w_obeta,st)
nd  = size(G,1);
    x_k.u       = [reshape(G,nd*nd,1);p;g_D_neg];
    x_k.w_obeta = w_obeta;
    x_k.st      = st;
end