function [nD]=NormDiffDistribution(X_i,X_j,Options)
% This function computes difference between distribution related to X_i and
% X_j based on hilbert space embedding of distributions
% This is very inefficient!
    N_i=X_i.NOS;
    N_j=X_j.NOS;
    S1=0;
    St=X_i.S;
    Sr=X_i.S;
    for t=1:N_i
        for r=1:N_i
            S1=S1+MyKernel(St{t},Sr{r},Options);
        end
    end
    S2=0;
    St=X_j.S;
    Sr=X_j.S;
    for t=1:N_j
        for r=1:N_j
            S2=S2+MyKernel(St{t},Sr{r},Options);
        end
    end
    SI=0;
    St=X_i.S;
    Sr=X_j.S;
    for t=1:N_i
        for r=1:N_j
            SI=SI+MyKernel(St{t},Sr{r},Options);
        end
    end
    nD=S1/(N_i*N_i)+S2/(N_j*N_j)-2*SI/(N_i*N_j);
    
end
function r=MyKernel(XSt,XSr,Options)
SigmainputSpace=Options.SigmaInInputSpace;
Sigma2 = SigmainputSpace^2 ;
r=exp(-norm(XSt-XSr)^2/(2*Sigma2));

end