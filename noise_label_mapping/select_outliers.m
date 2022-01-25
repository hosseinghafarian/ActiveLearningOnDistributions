function [ data, ind_outlier_select] = select_outliers(data, class_map, other_labels, n_select)


[ind_outlier_select] = select_a_subset_from_outlier_instances(numel(data.Y), n_select, other_labels);
n_cl            = size(class_map,1);
n_sel           = sum(ind_outlier_select);
rand_label      = class_map(randi(n_cl, n_sel, 1),2); % making random labels in our wanted labels
data.Y(ind_outlier_select) = rand_label;

end