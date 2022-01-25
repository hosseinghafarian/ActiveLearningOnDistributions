function Ind=k_mostsmallest(Array ,k)
    k = floor(k);
    n = numel(Array);
    if isempty(Array), Ind = []; return; end
    if k==0, Ind = []; return; end
    if k>=n, Ind = 1:n; return; end
    Arraycopy = Array;
    Ind   = zeros(k,1);
    for j = 1:k
       [a, Ind(j)] = min(Arraycopy);
       Arraycopy(Ind(j)) = +inf;
    end
    maximumValues = Array(Ind);
end