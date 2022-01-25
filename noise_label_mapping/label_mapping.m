function [data, ind_changed_lab] = label_mapping(data, class_map)
% Map each of classes to desired class, didn't add other instance from
% other classes if any. 

n_cl            = size(class_map,1);
n               = numel(data.Y);
ind_changed_lab = false(1,n);
ind_lb_i        = false(n_cl,n); 
for i =1:n_cl
    ind_lb_i(i,:)         = data.Y==class_map(i,1);
    ind_changed_lab  = ind_changed_lab | ind_lb_i(i,:); 
end

for i =1:n_cl
    data.Y(ind_lb_i(i,:)) = class_map(i,2);
    ind_changed_lab  = ind_changed_lab | ind_lb_i(i,:); 
end

end