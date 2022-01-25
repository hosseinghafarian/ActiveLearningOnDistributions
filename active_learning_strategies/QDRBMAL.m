function [query_id] = QDRBMAL(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
%This function implemented based on the paper: 
% Querying Discriminative and Representative Samples
% for Batch Mode Active Learning
% ZHENG WANG and JIEPING YE, Arizona State University
global cnstData

   xtrain = trainsamples.X;
   ytrain = trainsamples.Y;
   
   beta_Q = 1;
   lambda_Q = 0.1;
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   
   
   [ind]   = solveQDRBMAL(K, ytrain(Lindex)', Lindex, Uindex, Lind, Uind, beta_Q, lambda_Q);
   
   
   query_ind  = ind;
   query_id                                   = get_ID_fromind(trainsamples, query_ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in t(uhe distributions ID in learning data');
   if ismember(query_id, initL)
       query_id
   end
end
function [ind]   = solveQDRBMAL(K, y_l, Lindex, Uindex, Lind, Uind, beta_Q, lambda_Q)
global cnstData
   rho_ADMM    = 2;

   n_l         = sum(Lindex);
   n_u         = sum(Uindex);
   n           = n_l + n_u;
   b           = cnstData.batchSize;

   K_1         = 1/2*K(Uindex, Uindex);
   K_2         = (n_l + b)/n* sum(K(Uind, Uind), 2);
   K_3         = (n_u - b)/n* sum(K(Uind, Lind), 2);
   Kvec        = K_3 - K_2;
   converged   = false;
   query       = Uind(randperm(n_u, b));
   alpha       = zeros(n_u, 1);
   thau        = zeros(n_l+b,1);
   prethau     = thau;
   alpha(query)= 1; 
   while ( ~converged)
       thau  = computeWadmm(y_l, thau, alpha, query, K, Lindex, Uindex, rho_ADMM, lambda_Q);
       Ltil  = Lindex;
       Ltil (query) = true;
       alpha = computealphquad(Ltil, Uindex, K, K_1, Kvec, b, beta_Q, thau, n_l, n_u);
       td = norm(prethau-thau)/norm(thau);
       if( td <0.1)
           converged = true;
       end
       prethau = thau;
       qin_Uind = k_mostsmallest(1-alpha ,b);
       query    = Uind(qin_Uind);
       alpha(1:n_u) = 0;
       alpha(query) = 1;
   end    
  ind = query; 
end
function [thau]  = computeWadmm(y_l, thau_0, alpha, query, K, Lindex, Uindex, rho_ADMM, lambda_Q)
   Ltil   = Lindex;
   Ltil (query) = true;
   Qindex = false(numel(Uindex), 1);
   Qindex(query) = true;
   KLtilQ = K(Ltil, Qindex);
   KLtilL = K(Ltil, Lindex);
   
   eta_c = 2/(2+rho_ADMM);
   thau  = thau_0;
   gamma_t = 0;
   try 
      v     = (rho_ADMM*thau'*KLtilQ- gamma_t)/(rho_ADMM + 2);
   catch
      disp('hi');
   end
   z     = sign(v).*(abs(v)-eta_c); 
   for t=1:10
      r = y_l'*KLtilL' + 1/2*gamma_t*KLtilQ'+ rho_ADMM/2*z*KLtilQ';
      Bt = K(Ltil, Ltil);
      Cr = eye(size(Bt));
      A = (KLtilL* KLtilL') + rho_ADMM/2*(KLtilQ*KLtilQ')+ lambda_Q*Bt+ 0.001*Cr;
      thau = A\r';
      v     = (rho_ADMM*thau'*KLtilQ- gamma_t)/(rho_ADMM + 2);
      z = sign(v)*max(abs(v)-eta_c, 0);
      gamma_t = gamma_t + rho_ADMM*(z-thau'*KLtilQ);
   end
   
end
function [alpha_opt] = computealphquad(Ltil, Uindex, K, K_1, Kvec, b, beta_Q, thau, n_l, n_u)

Wphi = thau'*K(Ltil, Uindex);
a    = zeros(n_u, 1);
for j=1:sum(Uindex)
   a(j) = norm(Wphi(:,j))^2 + 2*abs(Wphi(:,j));
end
H  = beta_Q*K_1 ;
d  = beta_Q*Kvec + a;
alphavar = sdpvar(n_u, 1);
cConstraint = [ alphavar>=0 , alphavar<=1, sum(alphavar)==b];
cObjective = alphavar'*H*alphavar + d'*alphavar ;
ops = sdpsettings('verbose',0, 'solver', 'mosek');
sol = optimize(cConstraint, cObjective, ops);
if sol.problem == 0
    alpha_opt = value(alphavar);
else
    alpha_opt = -Inf;
end
end