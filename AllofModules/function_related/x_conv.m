function [x_k] = x_conv(G,p,w_obeta,st)
global cnstData
    x_k.u       = [reshape(G,cnstData.nSDP*cnstData.nSDP,1);p];
    x_k.w_obeta = w_obeta;
    x_k.st      = st;
end