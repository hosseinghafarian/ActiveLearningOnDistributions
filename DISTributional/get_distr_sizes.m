function [n_Dist] = get_distr_sizes(dataF)
dF         = diff(dataF);
idx        = [0, find(dF), numel(dataF)];
n_Dist = diff(idx);
end