function [optparams]  = optimaization_settings(learningparams)
global cnstData
    optparams.stmax_iter      = 100;
    optparams.stmax_iterADMM  = 30;
    optparams.maxouterprox    = 1000; 
    optparams.gtol            = 0.0001;  % start with a very large value for tolerance 
    optparams.alphatol        = optparams.gtol*10^-3;
    optparams.tol_ADMM        = 0.0001;
    optparams.tol4LinearSys   = 10^-7;
    optparams.maxit4LinearSys = 100;
    optparams.strongcvxmu     = learningparams.rhox;%0.0001; % strongly convex parameter
    optparams.thao            = 1;%1.618;% Best values for these parameters. Donot increase mul. , if these values changes in any iteration we must recompute cholesky factors below
    optparams.rhoml           = 1/(learningparams.lambda_o*learningparams.rhox+1);
    optparams.alphamaxit      = 100;
    optparams.mul             = 1;
end