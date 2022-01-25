function [K, K_o ]= get_two_KernelMatrix(X, Options)
% This is obsoloete. 
isDistributionalData = false;
if ~isDistributionalData
    switch Options.KernelType
        case 1
            K = X'*X;
            return
        case 2 % RBF Kernel, Sigma
            PD        = - pdist2(X',X');
            gamma     = Options.gamma;
            gamma_o = Options.gamma_o;
            K         = exp( gamma * PD );
            K_o       = exp( gamma_o * PD );
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