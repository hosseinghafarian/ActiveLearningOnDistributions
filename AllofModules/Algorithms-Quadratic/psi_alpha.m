function [fout,gout]                   = psi_alpha(alphav)
    global proxParam;
    global alphProx;
    global accumGradProx;
    fout = accumGradProx'*alphav + proxParam/2*norm(alphav-alphProx)^2;
    gout = accumGradProx         + proxParam  *(alphav-alphProx)      ;
end