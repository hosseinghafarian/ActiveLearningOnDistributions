function [K]= KernelMatrix(isDistributionalData,X,Options)

if ~isDistributionalData
    switch Options.KernelType
        case 1
            K = X'*X;
            return
        case 2 % RBF Kernel, Sigma
            Sigma2=Options.Sigmaexp2;
            K = exp(- pdist2(X',X')/(2*Sigma2));
            return 
    end
end
%% TODO: reexamine this part for Distributional data 
l = size(X,1);
% This part is very inefficient and must be boost it for performance, a
% code of paper by zoltan szabo which i tested and is part of ITE is very
% efficient 
for i=1:l
    for j=1:l
        K(i,j)=kernelDist(isDistributionalData,X(i),X(j),Options);
    end
end
end