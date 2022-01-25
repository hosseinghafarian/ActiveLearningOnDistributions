function [K]= getKernelMatrix(X,n,KOptions,reCompKernel)

kernelcfile='kernelcgaussianTest.mat';

if ~exist(kernelcfile, 'file') || reCompKernel
    K=KernelMatrix(X,n,KOptions);
    
    save(kernelcfile,'K');
else
    load(kernelcfile);
end
end