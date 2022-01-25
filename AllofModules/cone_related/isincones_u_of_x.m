function [is_in] = isincones_u_of_x(x_k)
global cnstData
nSDP  = cnstData.nSDP;
n_S   = cnstData.n_S;
G     = x_k.u(1:nSDP*nSDP);
G     = reshape(G,nSDP,nSDP);
G     = (G+G')/2;
is_sdp= isincone_sdp(G,nSDP);
p     = x_k.u( nSDP*nSDP+1:nSDP*nSDP+n_S);
is_Rplus = false;
if p>=0
    is_Rplus = true;
end
is_in = is_sdp & is_Rplus;
end