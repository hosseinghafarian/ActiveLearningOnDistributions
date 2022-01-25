function [Xapprox,p,q,qyu]             = getu_Parts(u)
    global cnstData
        %% previous :just for debug and observation
    Xapprox       = reshape(u(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
    p             = u(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);
    q             = Xapprox(cnstData.extendInd,cnstData.nSDP);
    qyu           = Xapprox(1:cnstData.n_S,cnstData.nSDP);
end