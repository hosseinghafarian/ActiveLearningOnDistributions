function [query_id] = noactivelearning(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData   
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);

   ind = Uind; 

   query_ind  = Uind(ind);
   query_id                                   = get_ID_fromind(trainsamples, query_ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end