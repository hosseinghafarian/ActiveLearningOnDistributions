function [data] = assume_others_as_outliers(data, other_labels, class_map) 
    n_cl            = size(class_map,1);
    n_other         = sum(other_labels);
    rand_label      = class_map(randi(n_cl, n_other, 1),2); % making random labels in our wanted labels
    data.Y(other_labels) = rand_label;
    data.noisy(other_labels) = true;
    data.labelnoise(other_labes) = false;
end