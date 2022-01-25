function [KA]= DISTKernelArray(model, xTest, F, learningparams)
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
reCompKA = true;
KAfile='KAfileComp.mat';
if reCompKA || ~exist(KAfile,'file')
        traindistidx = unique(model.F);
        ntr  = numel(traindistidx);
        testdistidx  = unique(F);
        nte  = numel(testdistidx);
        KA= zeros(numel(traindistidx),numel(testdistidx)); % I must boost this code for performance
        for t=1:ntr
            X_t  = model.vectrainx(:,model.F==traindistidx(t));
            for i=1:nte
                X_i  =     xTest(:,F==testdistidx(i));
                KA(t,i) =  exp(-0.5*learningparams.KOptions.gamma*NormDiffDistribution(X_t,X_i,learningparams.KOptions.gamma_is));
            end
        end
    save(KAfile,'KA');        
else % load KAfile 
    load(KAfile,'KA');
end
end