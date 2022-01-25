function [query_id] = ALFuncGradTikhonov(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   lambda = alparams{1};
   t_0    = alparams{2};
   mu     = alparams{3};      
   %xtrain = trainsamples.X;
   ytrain = trainsamples.Y; 
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);

   mu = 0;%no increase in boosting iterations for more labeled examples.
   
   [ind]       = ActiveFuncTikhonov_margin(K, Lind,Uind, ytrain(Lindex), lambda ,t_0 ,mu );


   query_id    = get_ID_fromind(trainsamples, ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end