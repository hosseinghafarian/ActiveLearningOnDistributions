function L          = computeLaplacianNormal(Kernel)
D  = sum(Kernel,2);
Dem = diag(D.^(-1/2)); 
L  = Dem*(diag(D)-Kernel)*Dem;
end