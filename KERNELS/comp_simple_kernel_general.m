function [TestSamples] = comp_simple_kernel_general(learningparams, Classification_exp_param, TestSamples, TrainSamples, uFTest, uFTrain) 

traintestequal = false;

if nargin<=4 
    uFTest  = unique(TestSamples.F);
    if nargin == 3
       uFTrain = uFTest;
       traintestequal = true;
    else
       uFTrain = unique(TrainSamples.F);
       traintestequal = false;
    end
end
if nargin==3
   [TestSamples.K, TestSamples.dm, TestSamples.F_to_ind_row, TestSamples.F_to_ind_col] = kernelArrayGeneral(TestSamples, uFTest, TestSamples, uFTest, learningparams, false); 
   TrainSamples = TestSamples;
else
   [TestSamples.K, TestSamples.dm, TestSamples.F_to_ind_row, TestSamples.F_to_ind_col] = kernelArrayGeneral(TrainSamples, uFTrain, TestSamples, uFTest, learningparams, false);    
   if traintestequal
       TrainSamples = TestSamples;
   end
end
[TestSamples.K, TestSamples.dm, TestSamples.F_to_ind_row, TestSamples.F_to_ind_col] = kernelArrayGeneral(TrainSamples, uFTrain, TestSamples, uFTest, learningparams, false);
if ~isempty(Classification_exp_param) && isfield(Classification_exp_param, 'two_kernel') && Classification_exp_param.two_kernel
    learningparam.use_secondkernel = true;
    [TestSamples.K_o, ~, ~] = kernelGeneral(learningparam, data);
end
end    