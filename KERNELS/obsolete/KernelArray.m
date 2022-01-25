function [KA]= KernelArray(isDistributionalData,xTrain,xTest,Options,reCompKA)
% Kernel function values Between Train and Test data
% isDistributionalData : is data a distribution or vectorial data
% xTrain : Train samples
% xTest  : Test  samples
% Options: Option for computing Kernel
% reCompKA : recompute kernel data file or use previous file stored.
if isempty(xTest)
   display('error Test Set is Empty');
   KA = xTest;
   return 
end
KAfile='KAfileComp.mat';
if reCompKA || ~exist(KAfile,'file')
    if ~isDistributionalData
        switch Options.KernelType
            case 1
                KA = xTrain'*xTest;
            case 2 
                Sigma2 = Options.Sigmaexp2;
                KA = exp(- pdist2(xTrain',xTest')/(2*Sigma2)) ;
        end
        save(KAfile,'KA');
        return 
    else %% if data is distributional
        %% TODO: re-examine this code
        l = size(xTrain,1);
        N = size(xTest,1);
        KA= zeros(N,l); % I must boost this code for performance
        for t=1:N
            for i=1:l
                KA(t,i) = kernelDist(isDistributionalData,xTrain(i),xTest(t),Options);
            end
        end
    end
    save(KAfile,'KA');        
else % load KAfile 
    load(KAfile,'KA');
end
end