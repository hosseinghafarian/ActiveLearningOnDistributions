function [alpha_next,iterAlpha,psialpha]= argminpsi(wmstalpha,alph0,accumGrad,accumf,rho,A_curr,lo,up,tol,maxit)
yalsolok  = 0;
global    proxParam;
global    alphProx;
global    accumGradProx;

accumGradProx = accumGrad;
proxParam     = (1+A_curr*rho);
alphProx      = alph0;

if yalsolok==0
    [x,histout,costdata,iterAlpha] = projbfgs(wmstalpha,@psi_alpha,up,lo,tol,maxit); % This is two order of magnitude faster than projected gradient
    %[x,histout,costdata,iterAlpha] = gradproj(wmstalpha,@psi_alpha,up,lo,tol,maxit);
    alpha_next = x;
    psialpha = accumf+accumGrad'*alpha_next;
elseif yalsolok==1 
    n    = size(alph0,1);
    alph = sdpvar(n,1);
    cObjective = accumGrad'*alph+ proxParam /2*norm(alph-alphProx)^2;
    cConstraint= [alph>=lo,alph<=up];
    ops = sdpsettings('verbose',0);
    sol = optimize(cConstraint,cObjective,ops);
    if sol.problem == 0
        alpha_next = value(alph);
        iterAlpha=2000;
    else
    
    end
end
end
