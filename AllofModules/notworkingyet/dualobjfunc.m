function [dobj]                        = dualobjfunc(rho,accumf,accumgradfalph,accumgrad,A_k,alph0,alpha_knew)
 
    dobj = accumf - accumgradfalph+...   % a_k f(\alpha_k)-\gradf(alpha_k)'*\alpha_k
           accumgrad'* alpha_knew+...   % (\sum_i \grad(f(\alpha_i))'* \alpha_knew
           A_k*rho/2*norm(alpha_knew-alph0)^2;
    dobj = dobj /A_k;
end