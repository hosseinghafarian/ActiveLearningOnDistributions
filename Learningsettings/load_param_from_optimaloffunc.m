function [lp_list] = load_param_from_optimaloffunc(load_opt_file, max_lp)
% This function loads the top three optimal learning parameters. 
load(load_opt_file, 'savedlp');
n_lp = numel(savedlp);
acc = zeros(n_lp, 1);
for i = 1:n_lp
    acc(i) = savedlp{i}.testmeasures.accuracy;
end
[ ~, ind] = sort(acc, 'descend');
n_top   = min(max_lp, numel(ind));
lp_list =  savedlp(ind(1:n_top));
end