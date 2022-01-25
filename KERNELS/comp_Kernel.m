function [ K ] = comp_Kernel(dm, gamma)
K   = exp(-0.5*gamma*dm);
end