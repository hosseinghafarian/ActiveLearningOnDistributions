function [query_id] = guss_harm_semi_al_wrapper(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   ytrain = trainsamples.Y; 
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);

   ytrain_col                = ytrain';
   nqueries                  = cnstData.batchSize;
   Y                         = [ytrain_col==-1,ytrain_col==1];
   [ind, acc_ML, risks]      = gaussian_harmonic_semisupervised_activelearning(nqueries, Y, K, Lind, Uind);

%    query_ind  = Uind(ind);
   query_id                                   = get_ID_fromind(trainsamples, ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end