function [s]                           = proj_oncones(s,nSDP,n_S,star)
G = s(1:nSDP*nSDP);
G = reshape(G,nSDP,nSDP);
G = (G+G')/2;
G = proj_sdp(G,nSDP);
p = s(nSDP*nSDP+1:nSDP*nSDP+n_S);

if star == 0
    p = max(p,0);                    % project on R+
else
    p = max(p,0);
end
s        = [reshape(G,nSDP*nSDP,1);p];
end