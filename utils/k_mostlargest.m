function Ind=k_mostlargest(Array ,k)
    k = floor(k);
    if k==0, Ind = []; return; end
    Arraycopy = Array;
    for j = 1:k
       [a, Ind(j)] = max(Arraycopy);
       Arraycopy(Ind(j)) = -inf;
    end
    maximumValues = Array(Ind);
end