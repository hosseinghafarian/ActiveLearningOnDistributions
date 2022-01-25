function [query_id] = DISTquireactive(trainsamples, initL, samples_to_query_from, K, lambda_AL, learningparams, al_par)
% In DISTquireactive we must use average in input domain, i.e. use
% trainsamples.DISTK. If we use K, it is equal to hilbert space embedding 
global cnstDefs
    ytrain = trainsamples.Y;
    K      = trainsamples.DISTK;
    [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
    y_Lindex   = ytrain(Lindex);
    lambda_AL         = learningparams.lambda_AL;
    [queryind] = QUIRE(K, Lind, Uind, y_Lindex, lambda_AL, false, trainsamples.invDISTKlam);

   query_id                                   = get_ID_fromind(trainsamples, queryind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end