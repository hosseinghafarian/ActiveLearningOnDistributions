function [queryind] = DISTrandomsampling(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, al_par) 
    Uindex    = setdiff ( samples_to_query_from, initL);
    nu        = numel(Uindex);
    q         = randi(nu);
    queryind  = Uindex(q); 
end