function [KA]= KernelArrayNewInstance(isDistributionalData,X,xnewTest,Options)
    
    l=size(X,1);
    N = size(xnewTest,1);
    KA=cell(N,1); % I must boost this code for performance
    for t=1:N
        for i=1:l
            KA{t}(i,1)=kernelDist(isDistributionalData,X(i),xnewTest,Options);
        end
    end
end