function [TestSamples] = comp_seriesappend_kernel(learningparams, Classification_exp_param, TestSamples, TrainSamples, uFTest, uFTrain) 
if nargin==4 
    uFTest  = unique(TestSamples.F);
    uFTrain = unique(TrainSamples.F);
end
[TestSamples.K, TestSamples.dm, TestSamples.F_to_ind_row, TestSamples.F_to_ind_col] = kernelArrayGeneral(TrainSamples, uFTrain, TestSamples, uFTest, learningparams, false);
if isfield(TrainSamples,'X_pcaappend') && isfield(TestSamples, 'X_pcaappend')
   idxTest  = 1:numel(TestSamples.F);
   idxTrain = 1:numel(TrainSamples.F);
   [TestSamples.K_L2, TestSamples.dm_L2, TestSamples.F_to_ind_row_L2] = kernelArrayGeneral(TrainSamples, idxTrain, TestSamples, idxTest, learningparam.L2_lp_init, true); 
end  

end    