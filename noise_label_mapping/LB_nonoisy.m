function [data] = LB_nonoisy(data, LB_settings)
% Map each of classes to desired class, didn't add other instance from
% other classes if any. 
class_map       = LB_settings.class_map;
if isthereanyotherlabel(data, class_map(:,2)) 
    [data,ind_changed_lab]     = label_mapping(data, class_map);
    data.X          = data.X(ind_changed_lab);
    data.Y          = data.Y(ind_changed_lab);
end
data.noisy      = false(1,numel(data.Y));     % there is no noisy instance(outlier or label noisy) here
data.labelnoise = false(1,numel(data.noisy)); % there is no label noise introduced here
data.n          = numel(data.Y);
end