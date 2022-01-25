function [KOptions ] = data_specific_settings(X, Z, Ktype)
%not used in any file: Obsolete. 
KOptions.KernelType        =   Ktype; % 1: Linear, 2: RBF, 
distpoints                 = pdist2(X',X');
refmed                     = median(median(distpoints))^2;
KOptions.Sigmaexp2         = refmed/2;% TODO: Verify Kernel function for Embedding Space
KOptions.gamma             = 2/refmed;
end