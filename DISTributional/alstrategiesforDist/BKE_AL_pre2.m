function [query_id] = BKE_AL(trainsamples, initL, samples_to_query_from, K, lambda, learningparams) 
global cnstData
   lambda_bu  = learningparams.lambda;
   sigmahat2  = learningparams.sigma_likelihood^2;

   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   
   ytrain     = trainsamples.Y; 
   cnstData.Uindex = Uindex;% What is this for? where we used it?
   
   y_Lindex   = ytrain(Lindex);
   try
      z_l        = (cnstData.BayesK(Lindex,Lindex)+eye(n_l)*lambda_bu)\y_Lindex'; %z_l        = inv(cnstData.BayesK(Lindex,Lindex)+eye(numel(Lindex))*sigma_wst)*label;
   catch
      n_l
   end
   KB_lu_z_l  = cnstData.BayesK(Uindex,Lindex)*z_l;
   call_mex_file = false;
   if call_mex_file
       normw_estimate   = w_lnorm_estimate(cnstData.BayesK(Lindex,Lindex), cnstData.BayesK(Uindex,Uindex) , KB_lu_z_l, ...
                                     cnstData.R, trainsamples.F, trainsamples.F_id, cnstData.mu_R_alli, cnstData.R_reginv_alli, ...
                                     initL, z_l);
   else
       normw_estimate   = w_lnorm_estimatem(cnstData.BayesK(Lindex,Lindex), cnstData.BayesK(Uindex,Uindex) , KB_lu_z_l, ...
                                     cnstData.R, trainsamples.F, trainsamples.F_id, cnstData.mu_R_alli, cnstData.R_reginv_alli, ...
                                     initL, z_l);                             
   end
   assert(sum(normw_estimate<=0)==0, 'ERROR: norm cannot be negative');
   Upsilon    = 1./(sigmahat2 + normw_estimate);                          
   sigma_wst  = lambda;
   U = Upsilon_estimate(trainsamples, y_Lindex, K, Lindex, Uindex, sigma_wst, learningparams.varrho);

   %Upsilon    = min(Upsilon, U');
   Uent       = entropy(U);
   Upsent     = entropy(Upsilon);
   
   Upsilon    = max(Upsilon, U');
   
   try
       [queryind] = BU_AL(cnstData.BayesK, Lind, Uind, y_Lindex, lambda_bu, Upsilon);
       query_id   = get_ID_fromind(trainsamples, queryind);
   catch
       query_id
   end
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end
function U = Upsilon_estimate(trainsamples, y_Lindex, K, Lindex, Uindex, sigma_wst, varrho)
   
%    invK_l     = inv(K(Lindex,Lindex)+eye(n_l)*sigma_wst);
%    normw2     = y_Lindex*invK_l*K(Lindex,Lindex)*invK_l*y_Lindex';
   
   H          = inv(K + eye(size(K))*sigma_wst);
   %Huuinv     = inv(H(Uindex,Uindex));
   h          = H(Uindex,Uindex)\ (H(Uindex, Lindex)*y_Lindex');
   normw2     = y_Lindex*H(Lindex,Lindex)*y_Lindex'-y_Lindex*H(Lindex, Uindex)*h;
   [n_Dist]   = get_distr_sizes(trainsamples.F);
   sigma_i2   = (varrho)./(n_Dist);
   U          = 1./(1+sigma_i2*normw2);%sigma_wstart^(-1)./(1+sigma_wstart^(-1)*sigma_i2*normw2);
end