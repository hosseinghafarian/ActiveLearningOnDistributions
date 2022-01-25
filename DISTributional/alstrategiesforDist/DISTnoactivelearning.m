function [queryind] = DISTnoactivelearning(xtrain, ytrain, initL, samples_to_query_from, K, lambda, learningparams)
    Uindex = setdiff ( samples_to_query_from, initL);
    queryind = Uindex; 
end