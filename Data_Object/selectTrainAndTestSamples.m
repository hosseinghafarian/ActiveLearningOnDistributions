function [TrainSamples, TestSamples] = selectTrainAndTestSamples(data, myprofile, TR_subsampling_method, TR_subsampling_settings)
% select a train and test samples from data
% output: labels of data as a column vector
TrainRatio = myprofile.trainratio; TestRatio = myprofile.testratio; Transductive = myprofile.Options.Transductive;
N        = data.n; % size of data 

if isfield(data, 'traintest_seperate') && data.traintest_seperate
    TestSamples  = data;
    TestSamples.X= data.test_X; 
    TestSamples.F= data.test_F;
    TestSamples.Y= data.test_Y; % Testindex is sorted!
    TestSamples.noisy      = false(1,numel(data.test_Y));
    TestSamples.labelnoise = false(1,numel(data.test_Y));
    TestSamples.n          = numel(TestSamples.Y);
    TrainSamples = data;    
    return;
end    
%jsut for test
if myprofile.dontshuffle && Transductive % don't shuffle the data 
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

divide_func = data.divide_func;

[TrainSamples, TestSamples] = divide_func(data, TrainRatio, TestRatio, TR_subsampling_method, TR_subsampling_settings);
end