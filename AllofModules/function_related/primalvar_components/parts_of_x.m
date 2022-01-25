function [Xapprox,p,w_obeta,st,q,qyu]  = parts_of_x(x_curr)
    global cnstData
        %% previous :just for debug and observation
    [Xapprox,p,q,qyu]     = getu_Parts(x_curr.u);
    w_obeta       = x_curr.w_obeta;
    st            = x_curr.st;
end