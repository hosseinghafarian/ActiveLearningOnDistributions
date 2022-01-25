function [KA, KA_o, F_to_ind_row, F_to_ind_col]= get_two_KernelArray(xtrain, xtest, learningparams, iseq)
% Kernel function values Between Train and Test data
% isDistributionalData : is data a distribution or vectorial data
% xTrain : Train samples
% xTest  : Test  samples
% Options: Option for computing Kernel
% reCompKA : recompute kernel data file or use previous file stored.
if isempty(xtest)
   display('error Test Set is Empty');
   KA = xtest;
   return 
end
if ~xtrain.isDistData
    gamma_o = learningparams.KOptions.gamma;
    [KA, dm, F_to_ind_row, F_to_ind_col] = kernelArrayGeneral(xtrain, [], xtest, [], learningparams, iseq);
    KA_o = exp(-0.5*gamma_o*dm);
else %% if data is distributional
    [KA, dm, F_to_ind_row, F_to_ind_col] = kernelArrayGeneral(xtrain, [], xtest, [], learningparams, iseq);
    KA_o  = exp(-0.5*learningparams.KOptions.gamma_o*dm);%comp_KernelEmb(dm, learningparams.KOptions.gamma_o);
end
end
%     F_to_ind_row = xtrain.F;
%     F_to_ind_col = xtest.F;
%     switch learningparams.KOptions.KernelType
%         case 1
%             KA = xtrain.X'*xtest.X;
%         case 2 
%             gamma    = learningparams.KOptions.gamma;
%             PD       = - pdist2(xtrain.X',xtest.X');
%             KA       = exp(0.5*gamma*PD) ;
%             gamma_o  = learningparams.KOptions.gamma_o;
%             KA_o     = exp(0.5* gamma_o * PD ) ;
%    end
