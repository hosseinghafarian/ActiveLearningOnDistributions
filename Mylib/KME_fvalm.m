function val = KME_fvalm(A, B, gamma)

[d,n] = size(A);
val = 0;
for i=1:n
    val = val + exp(-0.5*gamma*norm(A(:,i)-B(:,1))^2);
end
val = val / n;
end