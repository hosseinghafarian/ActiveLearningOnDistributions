function [copydata, ind_nonoisy_lab, not_selected] = label_mapping_witheachclasssubsample(data, labels_num, class_map )
n_cl            = size(class_map,1);
copydata        = data;
ind_nonoisy_lab = false(1,numel(data.Y));
not_selected    = false(1,numel(data.Y));
for i =1:n_cl
    ind_lb_i             = data.Y==class_map(i,1);
    % find instances from class class_map(i,1)
    ind_lb_sel           = find(ind_lb_i);
    % select the desired instances from each class
    randind              = randperm(numel(ind_lb_sel),labels_num(i));
    sel                  = ind_lb_sel(randind);
    ind_lb_sel_l         = false(1,data.n);
    ind_lb_sel_l(sel)    = true;
    % map the copied instances labels to class_map(i,2)
    copydata.Y(ind_lb_sel_l) = class_map(i,2);
    % update ind_nonnoisy_lab and not_selected lists
    ind_nonoisy_lab      = ind_nonoisy_lab | ind_lb_sel_l; 
    not_selected(setdiff(ind_lb_sel,sel))= true;
end
end