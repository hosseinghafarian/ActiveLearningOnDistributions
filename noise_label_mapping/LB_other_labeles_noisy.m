function [data] = LB_other_labeles_noisy(data, LB_settings)
% Map each of classes to desired class, adds all of instances from other classes
% as outlier instances with random labels 
data.n          = numel(data.Y);
data.noisy      = false(data.n,1);     % there is no noisy instance(outlier or label noisy) here
data.labelnoise = false(data.n,1); % there is no label noise introduced here

class_map       = LB_settings.class_map;
[data, ind_changed_lab] = label_mapping(data, class_map);

others    = ~ind_changed_lab;
[data]    = assume_others_as_outliers(data, others, class_map);

% add label noise if necessary
[data ]         = add_label_noise(data, LB_settings.label_noise_percent);
end