function [query_id] = DISTminimax_SVMAL(trainsamples, initL, samples_to_query_from, K, lambda, learningparams) 
% This function is based on paper: semisupervised SVM Batch Mode active
% learning with applications image retrieval. 
global cnstData
global cnstDefs

   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   % obtain Ktilde 
   xtrain  = trainsamples.DISTX;
   ytrain  = trainsamples.Y;
   K       = trainsamples.DISTK;
   y_Lindex= ytrain(Lindex);
   n                     = size(K,1);
   n_q                   = numel(samples_to_query_from);
   M                     = diag((K)*ones(n,1))-cnstData.K;
   Ktilde                = K-K*inv(eye(n)+K*M)*M*K;
   % obtain abs(f^*)

      
   [model]               = svmtrainwrapper_precomputed_kernel(0,  0, learningparams, xtrain(:,Lindex), y_Lindex, Ktilde(Lindex,Lindex));
   [predict_label, accuracy, decision_values] = svmpredictwrapper_precomputed_kernel(xtrain(:,Uindex), ytrain(Uindex), model, Ktilde(Uindex,Lindex));
   ftilde_abs_val                             = abs(decision_values);
   H        = learningparams.lambda*Ktilde(Uindex, Uindex);
   A_eq     = ones(1, n_q); b_eq = cnstData.batchSize;
   H        = (H+H')/2;
   H        = proj_sdp(H,n_q);
   q_var    = sdpvar(n_q,1);
   cObjec   = q_var'*ftilde_abs_val + 1/2* q_var'*H*q_var;
   cConstr  = [A_eq*q_var == b_eq, q_var>=0];
   %[q,fv,ef]= quadprog(H, ftilde_abs_val, [], [],A_eq, b_eq, zeros(n_q, 1)); 
   opts = sdpsettings('verbose', cnstDefs.solver_verbose);
   sol      = optimize(cConstr, cObjec, opts);
   if sol.problem == 0
       q    = value(q_var);
       ind      = k_mostsmallest(max(q, 0) ,cnstData.batchSize);
       queryind = Uind(ind);
   end
   
   query_id   = get_ID_fromind(trainsamples, queryind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in the distributions ID in learning data');
end