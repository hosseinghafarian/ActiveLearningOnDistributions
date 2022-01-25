function [query_id] = quireLAPLACIAN(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   %xtrain = trainsamples.X;
   ytrain = trainsamples.Y; 
   
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   y_Lindex   = ytrain(Lindex);

   [queryind] = QUIRE(K,Lind,Uind, y_Lindex, lambda );
   
   %query_id = trainsamples.F_id(queryind);
   query_id                                   = get_ID_fromind(trainsamples, queryind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end