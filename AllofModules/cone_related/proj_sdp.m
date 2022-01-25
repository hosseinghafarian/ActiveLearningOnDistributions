function z                             = proj_sdp(z,n)
eigtol = 1e-10;
if n==0
    return;
elseif n==1
    z = max(z,0);
    return;
end
try 
   [V,S] = eig(z);
catch
    try 
       warning('Rounding done in routine proj_sdp');
       nz     = sum(sum(z.^2));
       nz     = nz/numel(z);
       z(z.^2< eigtol*nz)=0;
       [V,S]  = eig(z);
    catch 
       save('matrix_eig_error','z','n'); 
       error('Problem using Proj_sdp routine, even with smoothing matrix, halting operation, matrix saved in file matrix_eig_error.mat');
    end
end
S = diag(S);

idx = find(S>0);
V = V(:,idx);
S = S(idx);
z = V*diag(S)*V';
end