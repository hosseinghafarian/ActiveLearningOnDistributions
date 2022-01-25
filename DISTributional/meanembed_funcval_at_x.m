function [val] = meanembed_funcval_at_x(gamma, nA, A, x)
val = 0;
for i=1:nA
    val = val + exp(-0.5*gamma* norm(A(:,i)-x)^2);
end
if val<1e-12
    val = 0;
end
val = val /nA;
end