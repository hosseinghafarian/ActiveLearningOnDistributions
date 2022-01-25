function [query_id] = uncertaintysamplingwrapper(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   xtrain = trainsamples.X;
   ytrain = trainsamples.Y;  
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);

   ind        = uncertainty_sampling(xtrain, ytrain,Lind, Uind);

   query_id   = get_ID_fromind(trainsamples, ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end