function value = x_norm(x_k,Q)
global cnstData
if nargin==1
    value = norm(x_k.u)^2+norm(x_k.st)^2+x_k.w_obeta'*cnstData.Q*x_k.w_obeta;
else
    value = norm(x_k.u)^2+norm(x_k.st)^2+x_k.w_obeta'*Q*x_k.w_obeta;
end
end