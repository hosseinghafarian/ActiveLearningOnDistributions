function [TestSamples] = comp_threeway_kernel(learningparams, Classification_exp_param, TestSamples, TrainSamples, uFTest, uFTrain, traintestequal) 
if nargin<=6 || nargin == 3;
    traintestequal = false;
end
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

n_tr   = numel(uFTrain);
n_te   = size(TestSamples.K, 2);       
K_3way = zeros(n_tr, n_te);
B           = learningparams.KOptions.B;
b2 = 2*B;
ind = 1:n_tr;
for i=1:n_tr
    for j=1:n_te 
       K_3way(i, j) = TestSamples.K(i,j)*exp(-norm(TrainSamples.K(i,ind)-TestSamples.K(ind, j)')^2/b2);
    end 
end
if n_tr==n_te 
   TestSamples.K_3way = Project_SD(K_3way);
else
   TestSamples.K_3way = K_3way; 
end
end