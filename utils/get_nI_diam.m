function [n_vec, IDX, D] = get_nI_diam(X, r)
% this function computes the number of instances in hypersphere in a
% diameter r of each point. 
[coeff, scoresX , latent] = pca(X');
X        = scoresX';
[IDX, D] = rangesearch(X', X', r);
n_idx    = size(X,2);
n_vec    = zeros(n_idx,1);
for i=1:n_idx
    n_vec(i) = numel(IDX{i});
end
end