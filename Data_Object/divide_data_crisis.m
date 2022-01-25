function [first_subset, second_subset] = divide_data_crisis(data, first_ratio, second_ratio, TR_subsampling_method, TR_subsampling_settings)
% output: labels of data as a column vector
N           = data.n; % size of data 
assert(first_ratio+ second_ratio <=1, 'sum of split data ration cannot be more than one in divide_data_crisis');
% select disjoint sets for sample and test data 

NTest       = floor(N*second_ratio);
NTT         = N;
% select non_noisy instances for test
n_noise     = sum(data.noisy);
nonoise_ind = find(~data.noisy);
Ti          = randperm(NTT-n_noise, NTest); % which samples from TTindex are for Test?
Testnumid   = sort(nonoise_ind(Ti));


reminder    = setdiff(nonoise_ind, Testnumid);
nr          = numel(reminder);
NTrain     = min(floor(first_ratio*(nr + NTest)), nr); 

Tri         = randperm(nr, NTrain);
Trainnumid  = sort(reminder(Tri));
[first_subset, second_subset] = divide_crisis_data_by_Fid(data, Trainnumid, Testnumid, TR_subsampling_method, TR_subsampling_settings);
end