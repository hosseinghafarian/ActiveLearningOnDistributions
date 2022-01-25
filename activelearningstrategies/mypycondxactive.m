function [query_id] = mypycondxactive(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   %xtrain = trainsamples.X;
   ytrain = trainsamples.Y; 
   
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   
   
   [query_ind] = mypycondx(K,Lind,Uind, ytrain(Lindex), lambda );
   
   %query_ind  = Uind(queryind);
   query_id                                   = get_ID_fromind(trainsamples, query_ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end