function [copydata] = LB_balanced_labeled(data, LB_settings)
% This function finds the two most frequent classes and map them to the
% desired classes, LB_settings.map_to_labels. Furthurmore, it balances labeled samples. Also, it downsamples instances with other labels to 
% the desired level, percent, i.e., LB_settings.down_sample_others_percent and the assign a random labels to these
% instances. 
data.n          = numel(data.Y);
data.noisy      = false(data.n,1);     % there is no noisy instance(outlier or label noisy) here
data.labelnoise = false(data.n,1); % there is no label noise introduced here

unbalanced_percent_tolerance = LB_settings.unbalanced_percent_tolerance;
percent_label_noise          = LB_settings.label_noise_percent;
lab             = LB_settings.map_to_labels;
percent         = LB_settings.down_sample_others_percent;
% find most two frequent labels
[numofclasses, lab_1, lab_2, max_lab_1, max_lab_2] = find_twomostfrequent(data);

n          = numel(data.Y);
% select label balance number
if max_lab_1 > max_lab_2 + floor(unbalanced_percent_tolerance*max_lab_2/100)
    max_lab_1 = max_lab_2 + floor(unbalanced_percent_tolerance*max_lab_2/100);
end
labels_num      = [max_lab_1; max_lab_2];
% map to desired classes and sample from each class
class_map       = [lab_1, lab(1);lab_2, lab(2)];
[copydata, ind_nonoisy_lab, not_selected] = label_mapping_witheachclasssubsample(data, labels_num, class_map );


other_labels    = ~ind_nonoisy_lab & ~not_selected;
n_nonnoisy      = sum(ind_nonoisy_lab);
n_other         = sum(other_labels);
n_select        = floor(n_nonnoisy*percent/100);
[ data, ind_outlier_select] = select_outliers(data,class_map, other_labels, n_select); % choose only selected samples (non-noisy and noisy) 

ind_select      = ind_nonoisy_lab | ind_outlier_select;
copydata.X      = data.X(:, ind_select);
copydata.Y      = copydata.Y(ind_select);
copydata.noisy  = ind_outlier_select(ind_select);
%copydata.noisy  = reshape(copydata.noisy, numel(data.noisy),1);
copydata.n      = numel(copydata.Y);
copydata.labelnoise = false(1,numel(copydata.noisy)); % there is no label noise introduced here
%add label noise if percent_label_noise > 0
[copydata ]     = add_label_noise(copydata, LB_settings.label_noise_percent);
end