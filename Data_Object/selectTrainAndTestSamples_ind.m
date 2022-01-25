function [TrainSamples_ind, TestSamples_ind] = selectTrainAndTestSamples_ind(selType, data, TestRatio,TrainRatio,Transductive,TR_subsampling_method, TR_subsampling_settings)
% select a train and test samples from data
% output: labels of data as a column vector
N        = size(data.X,2); % size of data 
if selType ==1 && Transductive % don't shuffle the data 
    TrainSamples = data;
    TestSamples  = data;
    return
end
if Transductive % if it is transductive, select the same data for both train and test
    % select SampleRatio of data, ignore TestRatio  
    NTrain   = floor(N*TrainRatio);
    Trindex  = randperm(N,NTrain)';
    data.X              = data.X(:, Trindex);
    data.Y              = data.Y(Trindex);
    data.noisy          = data.noisy(Trindex);
    data.labelnoise     = data.labelnoise(Trindex);
    return;
end
% select disjoint sets for sample and test data 
NTrain      = floor(N*TrainRatio);
NTest       = floor(N*TestRatio);
NTT         = NTrain+NTest;
T_index     = randperm(N, NTT)';
% select non_noisy instances for test
n_noise     = sum(data.noisy);
nonoise_ind = find(~data.noisy);
Ti          = randperm(NTT-n_noise, NTest); % which samples from TTindex are for Test?
Testindex   = nonoise_ind(Ti);
% select test samples first
Trindex         = setdiff(T_index, Testindex);
TestSamples_ind = Testindex;
TrainSamples_ind= Trindex;
end