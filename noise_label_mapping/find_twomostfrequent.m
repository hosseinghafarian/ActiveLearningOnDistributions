function [numofclasses, lab_1, lab_2, max_lab_1, max_lab_2] = find_twomostfrequent(data)
labels_in_data  = unique(data.Y);
numofclasses    = numel(labels_in_data);
lab_1           = 1; lab_2 = -1;
max_lab_1       = 0;
max_lab_2       = 0;
lab_1           = Inf;
for l = labels_in_data
    ind_lab = data.Y==l;
    s       = sum(ind_lab);
    if s> max_lab_1
        lab_2     = lab_1;
        max_lab_2 = max_lab_1;
        max_lab_1 = s;
        lab_1     = l;
    elseif s > max_lab_2
        max_lab_2 = s;
        lab_2     = l;
    end
end
end