function [query_id] = mybayesUncert(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
global cnstData
   %xtrain    = trainsamples.X;
   lambda     = learningparams.lambda;
   ytrain     = trainsamples.Y; 
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   y_Lindex   = ytrain(Lindex);
   sigma_wst  = lambda;% sigma_wstart   is the ratio (sigma_w^{-2}/sigma^{-2}) 
   
   invK_l     = inv(K(Lindex,Lindex)+eye(n_l)*sigma_wst);
  
   normw2     = y_Lindex*invK_l*K(Lindex,Lindex)*invK_l*y_Lindex';
   sigma_i2   = 1./(cnstData.Neighbors);
   
   [ind]      = mybayesianUncertainAL(K, Lind, Uind, y_Lindex, lambda, normw2, sigma_i2, sigma_wst );
   
   query_ind  = Uind(ind);
   query_id                                   = get_ID_fromind(trainsamples, query_ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end