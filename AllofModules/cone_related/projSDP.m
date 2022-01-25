function S = projSDP(M)
[V,D]=eig(M);
D_p  =diag(D);
D_p  =bsxfun(@max,D_p,0);
S    =V'*diag(D_p)*V;
end