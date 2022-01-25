function [TestSamples] = comp_twolevel_kernel(learningparams, Classification_exp_param, TestSamples, TrainSamples, uFTest, uFTrain) 
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
if traintestequal
   TrainSamples = TestSamples;
end
[TestSamples.K, TestSamples.dm, TestSamples.F_to_ind_row, TestSamples.F_to_ind_col] = kernelArrayGeneral(TrainSamples, uFTrain, TestSamples, uFTest, learningparams, false);
if isfield(learningparams,'isTwoLevel') && learningparams.isTwoLevel
   idxTest  = 1:numel(TestSamples.F);
   idxTrain = 1:numel(TrainSamples.F);
   [TestSamples.K_L2, TestSamples.dm_L2, TestSamples.F_to_ind_row_L2] = kernelArrayGeneral(TrainSamples, idxTrain, TestSamples, idxTest, learningparams.L2_lp_init, true); 
   TestSamples.L = comp_combkernel(TestSamples, TrainSamples);
end  


   function [L ]  = comp_combkernel(TestSamples, TrainSamples)
        distidxtest  = 1:TestSamples.n;      
        testi        = 1:numel(TestSamples.F);
        
        distidxtrain = 1:TrainSamples.n;      
        traini       = 1:numel(TrainSamples.F);
        
        
        K            = TestSamples.K(distidxtrain, distidxtest);% Here we must update the kernel matrix if kernel params changed = data.K(idx,idx);
        K_L2         = TestSamples.K_L2(traini, testi);  
        L            = K_L2;
        n_disttest   = TestSamples.n;
        n_disttrain  = TrainSamples.n;
        for fidx     = 1:n_disttest
            testi_fidx = find(ismember(TestSamples.F, TestSamples.F_id(fidx)));
            for tidx  = 1:n_disttrain 
                traini_tidx = find(ismember(TrainSamples.F, TrainSamples.F_id(tidx)));
                for i=testi_fidx
                    for j= traini_tidx
                       L(j, i) = K(tidx, fidx)*K_L2(j, i);
                    end
                end    
            end    
        end    
    end
end    