function [R]                           = proj_onP(R,nSDP,n_S,query)
%P a set in which query elements of R, is positive. 
RMat    = reshape(R(1:nSDP*nSDP),nSDP,nSDP);
Rpquery = max(RMat(query,nSDP),0);
RMat(query,nSDP ) = Rpquery;
RMat(nSDP ,query) = Rpquery;
R(1:nSDP*nSDP)    = reshape(RMat,nSDP*nSDP,1);
end
