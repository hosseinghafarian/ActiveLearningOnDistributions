function [first_subset, second_subset] = divide_data_general(data, first_ratio, second_ratio, TR_subsampling_method, TR_subsampling_settings)
% output: labels of data as a column vector
N        = data.n; % size of data 
% select disjoint sets for sample and test data 
NTrain      = floor(N*first_ratio);
NTest       = floor(N*second_ratio);
NTT         = N;
% select non_noisy instances for test
n_noise     = sum(data.noisy);
nonoise_ind = find(~data.noisy);
Ti          = randperm(NTT-n_noise, NTest); % which samples from TTindex are for Test?
Testnumid   = sort(nonoise_ind(Ti));
Trainnumid  = setdiff(nonoise_ind, Testnumid);
datasplit = data.split_data_func;
[first_subset, second_subset] = datasplit(data, Trainnumid, Testnumid, TR_subsampling_method, TR_subsampling_settings);
end