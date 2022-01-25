function [x_k ] = project_u_of_x(x_k)
global cnstData
nSDP  = cnstData.nSDP;
n_S   = cnstData.n_S;
G     = x_k.u(1:nSDP*nSDP);
G     = reshape(G,nSDP,nSDP);
G     = (G+G')/2;
G     = proj_sdp(G,nSDP);
p     = x_k.u( nSDP*nSDP+1:nSDP*nSDP+n_S);
p     = max(p,0);
x_k.u = [reshape(G,nSDP*nSDP,1);p];
end