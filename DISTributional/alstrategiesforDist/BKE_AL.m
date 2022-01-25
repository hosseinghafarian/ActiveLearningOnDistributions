function [query_id] = BKE_AL(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, Almethod_params) 
global cnstData
   lambda_AL  = learningparams.lambda_AL;
   lambda_bu  = learningparams.lambda_AL;
   sigmahat2  = learningparams.sigma_likelihood^2;

   n          = trainsamples.n;
   ytrain     = trainsamples.Y; 
   KBr_inv    = inv(cnstData.BayesK+eye(n)*lambda_bu);
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);

   y_Lindex   = ytrain(Lindex);

   call_mex_file = true;
   tol        = 1e-10;
    if call_mex_file
       normw_est   = normw_estimate(y_Lindex', Lind, Uind, cnstData.BayesK, KBr_inv,...
                                     cnstData.R, trainsamples.F, trainsamples.F_id, cnstData.mu_R_alli, cnstData.R_reginv_alli, tol);
    else
       normw_est1   = normw_estimatem(y_Lindex', Lind, Uind, cnstData.BayesK, KBr_inv,...
                                     cnstData.R, trainsamples.F, trainsamples.F_id, cnstData.mu_R_alli, cnstData.R_reginv_alli, tol);                             
    end
%    if (sum(normw_est<=0)~=0)
%        disp('ERROR: norm cannot be negative');
%    end
   assert(sum(normw_est<=0)==0, 'ERROR: norm cannot be negative');

%    if(norm(normw_est1-normw_est)>1e-4)
%        disp('Error');
%    end
   Upsilon    = 1./(sigmahat2 + normw_est);    
%    sUps       = sum(Upsilon);
%    Upsilon    = Upsilon/sUps;

%    sigma_wst  = lambda;
%    U = Upsilon_estimate(trainsamples, y_Lindex, K, Lindex, Uindex, sigma_wst, learningparams.varrho);
% 
%    %Upsilon    = min(Upsilon, U');
%    Uent       = entropy(U);
%    Upsent     = entropy(Upsilon);
%    
%    Upsilon    = max(Upsilon, U');
   
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