function [query_id] = FSimp2CompL2Boost(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   t_simple  = alparams{1};
   t_complex = alparams{2};
   t_step    = alparams{3};
   t_range   = t_simple:t_step:t_complex;
   top_n     = alparams{4};
   lambda    = alparams{5};
   %xtrain = trainsamples.X;
   ytrain = trainsamples.Y; 
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);

   
   [ind] = FromSimple2ComplexAL(K, Lind,Uind, ytrain(Lindex), t_simple, t_complex, t_step, top_n, lambda );


   query_id                                   = get_ID_fromind(trainsamples, ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end