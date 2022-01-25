function [KA, dm, F_to_ind_row, F_to_ind_col] = kernelArrayGeneral(xtrain, Fidxtr, xtest, Fidxte, learningparams, iseq)
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
kernel_func = learningparams.KOptions.kernel_func;
gamma    = learningparams.KOptions.gamma;
if isfield(learningparams,'use_secondkernel') && learningparams.use_secondkernel 
    gamma = learningparams.KOptions.gamma_o;
end
  
[KA, dm, F_to_ind_row, F_to_ind_col] = kernel_func(xtrain, Fidxtr, xtest, Fidxte, learningparams, gamma, iseq);
% if ~xtrain.isDistData
%     F_to_ind_row = xtrain.F;
%     F_to_ind_col = xtest.F;
%     if isempty(Fidxtr)
%         Fidxtr = true(numel(F_to_ind_row),1);
%     end
%     if isempty(Fidxte)
%         Fidxte = true(numel(F_to_ind_col),1);
%     end
%     switch learningparams.KOptions.KernelType
%         case 1
%             KA = xtrain.X(:,Fidxtr)'*xtest.X(:,Fidxte);
%         case 2 
%             dm       = pdist2(xtrain.X(:,Fidxtr)',xtest.X(:,Fidxte)');
%             KA       =  comp_Kernel(dm, gamma) ;
%     end
%     return 
% else %% if data is distributional
%     [dm, F_to_ind_row, F_to_ind_col] = distance_matrix(xtrain, Fidxtr, xtest, Fidxte, learningparams.KOptions.gamma_is, iseq);
%     KA    = comp_Kernel(dm, gamma);
% end
end