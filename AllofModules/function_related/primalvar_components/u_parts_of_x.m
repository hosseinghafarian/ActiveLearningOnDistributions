function [Xapprox,p,q,qyu]             = u_parts_of_x(x_k)
    global cnstData
        %% previous :just for debug and observation
    Xapprox       = reshape(x_k.u(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
    p             = x_k.u(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);
    q             = Xapprox(cnstData.extendInd,cnstData.nSDP);
    qyu           = Xapprox(1:cnstData.n_S,cnstData.nSDP);
end