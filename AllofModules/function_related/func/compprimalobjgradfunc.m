function [pobj,gradobj]                = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_curr,x_G,p,q)
global cnstData

b       = ones(size(p))-p;
b(cnstData.unlabeled) = b(cnstData.unlabeled)-q;
a       = [b;zeros(cnstData.nSDP-cnstData.n_S-1,1)];
pobj    = -c_k'*x_curr.u+learningparams.rhox/2*norm(x_curr.u-x_G.u)^2+learningparams.rhox/2*norm(x_curr.st-x_G.st)^2 ...
              +learningparams.lambda_o/2*x_curr.w_obeta'*cnstData.K*x_curr.w_obeta...
              +learningparams.rhox/2*(x_curr.w_obeta-x_G.w_obeta)'*cnstData.Q*(x_curr.w_obeta-x_G.w_obeta);

uk      = cnstData.Kuvec.*x_curr.u;
UDOTK   = reshape(uk(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP); 
gradobj = 1/learningparams.lambda* UDOTK(1:cnstData.nSDP-1,1:cnstData.nSDP-1)*alpha_k-a;          
end