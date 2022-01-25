function R = R_forall(nA, A, gamma_r)
R = zeros(nA, nA);
for i=1:nA
    for j=1:i
        R(i,j) = exp(-0.5*gamma_r* norm(A(:,i)-A(:,j))^2); 
        R(j,i) = R(i,j); 
    end
end
end