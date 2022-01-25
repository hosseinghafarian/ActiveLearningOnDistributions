function [nD]=NormDiffDistribution(X_i,X_j, gamma)
% This function computes difference between distribution related to X_i and
% X_j based on hilbert space embedding of distributions
% This is very inefficient and must be optimized for better performance
    N_i=size(X_i,2);
    N_j=size(X_j,2);
    S1=0;
    for t=1:N_i
        for r=1:N_i
            S1=S1+MyKernel(X_i(:,t),X_i(:,r),gamma); %MyKernel(X_i(:,t),X_i(:,r),gamma_inputspace);
        end
    end
    S2=0;
    for t=1:N_j
        for r=1:N_j
            S2=S2+MyKernel(X_j(:,t),X_j(:,r),gamma);
        end
    end
    SI=0;
    for t=1:N_i
        for r=1:N_j
            SI=SI+MyKernel(X_i(:,t),X_j(:,r), gamma);
        end
    end
    nD=S1/(N_i*N_i)+S2/(N_j*N_j)-2*SI/(N_i*N_j);
end
function r=MyKernel(XSt,XSr,gamma)

r=exp(-0.5*gamma* norm(XSt-XSr)^2);
end