function [query_id] = DISTmaniadpat_exp_design(trainsamples, initL, samples_to_query_from, K, lambda, learningparams) 
global cnstData
    xtrain  = trainsamples.DISTX;
    ytrain  = trainsamples.Y;
    
    [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
    

    Options.splitLabel        = false(numel(ytrain),1);
    Options.splitLabel(Lind)  = true;
    Options.k                 = 5;
    Options.ReguBeta          = 0.1;
    Options.ReguAlpha         = 0.01;
    
    queryind   = MAED(xtrain, cnstData.batchSize, Options);
    query_id   = get_ID_fromind(trainsamples, queryind);
    assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end