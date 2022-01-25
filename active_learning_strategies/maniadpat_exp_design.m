function [query_id] = maniadpat_exp_design(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   xtrain = trainsamples.X;
   ytrain = trainsamples.Y; 
    [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);

    Options.splitLabel        = false(numel(ytrain([Lind,Uind])),1);
    Options.splitLabel(1:n_l) = true;
    Options.k                 = 5;
    Options.ReguBeta          = 0.1;
    Options.ReguAlpha         = 0.01;
    ind                       = MAED(xtrain(:,[Lind, Uind])', cnstData.batchSize, Options);

    query_ind  = Uind(ind);
    query_id                                   = get_ID_fromind(trainsamples, query_ind);
    assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end