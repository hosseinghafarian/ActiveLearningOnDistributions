function [data ] = add_label_noise(data, percent)
% This function add percent% label noise to data. That is it flips labels from -1 ,1 to the other  
%find instances which are not noisy, i.e. correct label instance
not_noisy           = ~data.noisy;
ind_notnoisy        = find(not_noisy);
%compute appropriate amount of label noise
n_notnoisy          = sum(not_noisy);
n_add_noise         = floor(n_notnoisy*percent/100);
%add label noise 
ind_rand            = randperm(n_notnoisy, n_add_noise);
ind_lbnoise         = ind_notnoisy(ind_rand);
data.Y(ind_lbnoise) = -1*data.Y(ind_lbnoise);
data.labelnoise(ind_lbnoise) = true;
data.noisy          = data.noisy | data.labelnoise;
end