function outvec = KME_fval_allm(A, gamma)

[d,n] = size(A);
for j=1:n
    val = 0;
    for i=1:n
        val = val + exp(-0.5*gamma*norm(A(:,i)-A(:,j))^2);
    end
    val = val / n;
    outvec(j) = val;
end
end