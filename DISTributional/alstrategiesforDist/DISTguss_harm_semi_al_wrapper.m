function [query_id] = DISTguss_harm_semi_al_wrapper(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, al_par)
% In DISTquireactive we must use average in input domain, i.e. use
% trainsamples.DISTK. If we use K, it is equal to hilbert space embedding
global cnstData
    xtrain  = trainsamples.DISTX;
    ytrain  = trainsamples.Y;
    K       = trainsamples.DISTK;
    
    [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
    
    
    ytrain_col                = ytrain';
    nqueries                  = cnstData.batchSize;
    Y                         = [ytrain_col==-1,ytrain_col==1];
    [queryind, acc_ML, risks] = DISTgaussian_harmonic_semisupervised_activelearning(nqueries, Y, K, Lind, Uind);
    
    query_id   = get_ID_fromind(trainsamples, queryind);
    assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');    
end