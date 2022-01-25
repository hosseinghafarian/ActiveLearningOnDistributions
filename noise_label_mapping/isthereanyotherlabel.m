function [yesthereis] = isthereanyotherlabel(data, labellist)
% Map each of classes to desired class, didn't add other instance from
% other classes if any. 
n_cl            = numel(labellist);
n               = numel(data.Y);
ind_changed_lab = false(1,n);
ind_lb_i        = false(n_cl,n); 
for i =1:n_cl
    ind_lb_i(i,:)         = data.Y==labellist(i);
    ind_changed_lab  = ind_changed_lab | ind_lb_i(i,:); 
end
yesthereis = sum(ind_changed_lab)~=n;
end