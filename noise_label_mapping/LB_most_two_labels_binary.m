function [data] = LB_most_two_labels_binary(data, LB_settings)
% This function finds the two most frequent classes and map them to the
% desired classes, LB_settings.map_to_labels. Furthurmore, it down samples, instances with other labels to 
% the desired level, percent, i.e., LB_settings.down_sample_others_percent and the assign a random labels to these
% instances. 
data.n          = numel(data.Y);
data.noisy      = false(data.n,1);     % there is no noisy instance(outlier or label noisy) here
data.labelnoise = false(data.n,1); % there is no label noise introduced here

lab             = LB_settings.map_to_labels;
percent         = LB_settings.down_sample_others_percent;
% find most two frequent labels
[numofclasses, lab_1, lab_2,~,~] = find_twomostfrequent(data);

% map to desired classes 
class_map       = [lab_1, lab(1);lab_2, lab(2)];
n_cl            = 2;
[data, ind_changed_lab] = label_mapping(data, class_map);

other_labels    = ~ind_changed_lab;
[ data, ind_outlier_select] = select_outliers(data, other_labels, n_nonnoisy);% choose only selected samples (non-noisy and noisy) 

ind_select      = ind_changed_lab | ind_outlier_select;
data.X          = data.X(:, ind_select);
data.Y          = data.Y(ind_select);
ind             = ind_noisy_select(ind_select);
data.noisy(ind) = true;
data.labelnoise(ind) = true;
% add label noise if necessary
[data ]         = add_label_noise(data, LB_settings.label_noise_percent);
end