function [pobj ]                       = primalobjfunc(learningparams,c_a,alpha_k,x_curr,x_G,p,q)
global cnstData
    [c_k]    = computecoeff(learningparams.lambda,c_a,alpha_k);
    [pobj,~] = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_curr,x_G,p,q);                           
end