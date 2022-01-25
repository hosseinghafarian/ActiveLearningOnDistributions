function [sigma] = median_heurisitc(X)
    Z  = X';
    size1=size(Z,1);
    if size1>5000
      ind  = sort(randperm(size1,5000));  
      Zmed = Z(ind,:);
      size1 = 5000;
    else
      Zmed = Z;
    end
    G = sum((Zmed.*Zmed),2);
    Q = repmat(G,1,size1);
    R = repmat(G',size1,1);
    dists = Q + R - 2*Zmed*Zmed';
    dists = dists-tril(dists);
    dists=reshape(dists,size1^2,1);
    sigma = sqrt(0.5*median(dists(dists>0)));  %rbf_dot has factor two in kernel
end