function [K_ij]= kernelDist(isDistributionalData,X_i,X_j,Options)
Sigma=Options.Sigma;
Sigma2=Sigma*Sigma;
if isDistributionalData
    K_ij=exp(-0.5*gamma*NormDiffDistribution(X_i,X_j,Options));
else % it is a normal kernel value computation with input gamma if it is not distributional
    switch Options.KernelType
        case 1
            K_ij = X_i'*X_j;
            return
        case 2 
            K_ij=exp(-0.5*gamma*norm(X_i-X_j));
            return 
    end
    
end

end