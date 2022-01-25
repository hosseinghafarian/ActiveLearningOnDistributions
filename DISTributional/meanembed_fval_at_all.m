function [meb_vec] = meanembed_fval_at_all(gamma, nA, A)
meb_vec = zeros(nA,1);
for i=1:nA
   meb_vec(i) = meanembed_funcval_at_x(gamma, nA, A, A(:,i));
end

end