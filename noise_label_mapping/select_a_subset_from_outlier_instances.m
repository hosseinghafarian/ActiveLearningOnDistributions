function [ind_noisy_select] = select_a_subset_from_outlier_instances(n, n_select, other_labels)

n_other         = sum(other_labels);
ind_noisy_select= false(1, n);
%select from other labels as noisy instances
noisy_ind       = find(other_labels);
if n_other > n_select
   other_labels_sel = randperm(n_other, n_select); 
else
   other_labels_sel = other_labels;
end
noisy_sel       = noisy_ind(other_labels_sel);
n_sel           = numel(noisy_sel);
ind_noisy_select(noisy_sel) = true;

end