function [query_id] = myHILBEMbayesUncert(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, al_par) 
global cnstData
   lambda_AL         = learningparams.lambda_AL;
   sigma_wst  = lambda_AL;%1;% sigma_wstart   is the ratio (sigma_w^{-2}/sigma^{-2}) 
   
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   
   ytrain     = trainsamples.Y; 
   invK_l     = inv(K(Lindex,Lindex)+eye(n_l)*sigma_wst);
   y_Lindex   = ytrain(Lindex);
   
   normw2     = y_Lindex*invK_l*K(Lindex,Lindex)*invK_l*y_Lindex';

   [n_Dist]   = get_distr_sizes(trainsamples.F);
   sigma_i2   = (learningparams.varrho)./(n_Dist);
   [queryind] = mybayesianUncertainAL(K, Lind, Uind, y_Lindex, lambda_AL, normw2, sigma_i2, sigma_wst );
   
   query_id   = get_ID_fromind(trainsamples, queryind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end