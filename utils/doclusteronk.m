function Kclustered = doclusteronk(K, ef_rho)
n = numel(ef_rho);
assert(size(K,1)==n,'Matrix and the number of instances doesnot match in doclusteronk');
Kclustered = zeros(n);
for i = 1:n
   n_i = numel(ef_rho{i});
   Kclustered(i,i) = K(i,i);
   for t=1:n_i
       j = ef_rho{i}(t);
       Kclustered(i,j) = K(i,j); 
       Kclustered(j,i) = K(i,j); 
   end
end
Kclustered = 1/2*(Kclustered + Kclustered');

end