function z = proj_sdp(y,n)
if n==0
    return;
elseif n==1
    y = max(y,0);
    return;
end

% expand to full size matrix
z = tril(ones(n));
%z(z == 1) = y; <-previously
z(z == 1) = y(z==1);
z = (z + z');
z = z - diag(diag(z)) / 2;


[V,S] = eig(z);
S = diag(S);

idx = find(S>0);
V = V(:,idx);
S = S(idx);
z = V*diag(S)*V';

%z = z(tril(ones(n)) == 1);
end