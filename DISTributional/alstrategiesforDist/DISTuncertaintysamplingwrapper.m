function [query_id] = DISTuncertaintysamplingwrapper(trainsamples, initL, samples_to_query_from, K, lambda, learningparams) 
   xtrain  = trainsamples.DISTX;
   ytrain  = trainsamples.Y;
   
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   
   queryind = DISTuncertainty_sampling(xtrain, ytrain, Lind, Uind);
   
   query_id   = get_ID_fromind(trainsamples, queryind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end