function [query_id] = BKE_AL(trainsamples, initL, samples_to_query_from, K, lambda, learningparams) 
global cnstData
   ytrain     = trainsamples.Y; 
   unlabeled  = setdiff ( samples_to_query_from, initL)';
   Uindex     = ismember(trainsamples.F_to_ind, unlabeled);
   initL      = initL(initL>0)';
   Lindex     = ismember(trainsamples.F_to_ind, initL);
   
   cnstData.Uindex = Uindex;% What is this for? where we used it?
   lambda_bu  = learningparams.lambda;
   sigmahat2  = learningparams.sigma_likelihood^2;
   
   label      = ytrain(Lindex);
   %invK_lhatB = inv(cnstData.BayesK(Lindex,Lindex)+eye(numel(Lindex))*sigma_wst);
   %z_l        = invK_lhatB*label;
   
   try
      z_l        = (cnstData.BayesK(Lindex,Lindex)+eye(n_l)*lambda_bu)\label';
   catch
      n_l
   end
   KB_lu_z_l  = cnstData.BayesK(Uindex,Lindex)*z_l;
   normwl_t   = w_lnorm_estimate(cnstData.BayesK(Lindex,Lindex), cnstData.BayesK(Uindex,Uindex) , KB_lu_z_l, ...
                                 cnstData.R, trainsamples.F, trainsamples.F_id, cnstData.mu_R_alli, cnstData.R_reginv_alli, ...
                                 initL, z_l);
%    normwl_t   = w_lnorm_estimatem(cnstData.BayesK(Lindex,Lindex), cnstData.BayesK(Uindex,Uindex) , KB_lu_z_l, ...
%                                  cnstData.R, trainsamples.F, trainsamples.F_id, cnstData.mu_R_alli, cnstData.R_reginv_alli, ...
%                                  initL, z_l);                             
   Upsilon    = 1./(sigmahat2 + normwl_t);                          
   Lind       = find(Lindex);
   Uind       = find(Uindex);
   try
       [queryind] = BU_AL(cnstData.BayesK, Lind, Uind, ytrain(Lindex), lambda_bu, Upsilon);
       query_id   = trainsamples.F_to_ind(queryind);
   catch
       query_id
   end
   if ~ismember(query_id, trainsamples.F_id)
       query_id
   end
end