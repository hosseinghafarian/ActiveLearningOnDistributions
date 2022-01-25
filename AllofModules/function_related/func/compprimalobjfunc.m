function [pobj]                        = compprimalobjfunc(arho,lambda_o,Xr,s,w_obeta,G,g,w_obetapre,K,c_k,Q)

pobj  = -c_k'*Xr+arho/2*norm(Xr-G)^2+arho/2*norm(s-g)^2 ...
              +lambda_o/2*w_obeta'*K*w_obeta+arho/2*(w_obeta-w_obetapre)'*Q*(w_obeta-w_obetapre);
end