function [query_id] = TcybERIAL_PrLIBSVM(trainsamples, initL, samples_to_query_from, K, lambda, learningparams, alparams) 
%This function implemented based on the paper: 
% Exploring Representativeness and Informativeness
% for Active Learning, 
% authors: Bo DU, Zengmao WANG, Lefei ZHANG, Liangpei ZHANG, Wei LIU, Jialie SHEN, and Dacheng TAO
% published in: IEEE Transactions on Cybernetics, 2017 Jan, vol. 47, no. 1, pp. 14-26
global cnstData
gammaERIAL = 1;
   xtrain = trainsamples.X;
   ytrain = trainsamples.Y;
   
   beta_ERIAL = 1;
   [n_l, Lindex, Uindex, Lind, Uind] = get_indices( trainsamples, samples_to_query_from, initL);
   
   Uall = find(Lindex | Uindex);
   Ufull = true(numel(Lindex),1);
   learningparams.probestimate  = true;
   [model]                                    = svmtrainwrapper(learningparams, trainsamples, Lind,xtrain(:,Lindex), ytrain(Lindex));
   SVind = Lind(model.sv_indices);
   SVindex = false(numel(Uall),1);
   SVindex(SVind) = true;
%   [predict_label, accuracy, decision_values] = svmpredictwrapper(model, learningparams, trainsamples, Uind, xtrain(:,Uindex), ytrain(Uindex));
   [predict_label, accuracy, decision_values] = svmpredictwrapper(model, learningparams, trainsamples, Ufull, xtrain(:,Ufull), ytrain(Ufull));
   probs = decision_values;
   [M1, M2, M3, C] = computeERIALmeasures(model, gammaERIAL, probs, Lindex, Uindex, SVindex);
   [alpha_ERIAL]   = solveERIALquad(M1, M2, M3, C, beta_ERIAL);
   abs_dec_val                                = abs(alpha_ERIAL);
   ind                                        = k_mostsmallest(1-abs_dec_val ,cnstData.batchSize);
   
   query_ind  = Uind(ind);
   query_id                                   = get_ID_fromind(trainsamples, query_ind);
   assert(ismember(query_id, trainsamples.F_id), 'Error: query_id is not in t(uhe distributions ID in learning data');
   if ismember(query_id, initL)
       query_id
   end
end
function [M1, M2, M3, C] = computeERIALmeasures(model, gammaERIAL, probs, Lindex, Uindex, SVindex)
n_t   = sum(Lindex);
u_t   = sum(Uindex);
n     = n_t + u_t;
nall  = numel(Lindex);
nfull = size(probs, 1);
S_lab = zeros(1, nfull);
S_lab(Lindex) = 1;
S_unlab = zeros(1, nfull);
S_unlab(Uindex) = 1;

D     = pdist2(probs, probs);
Mall  = 1/2* exp(-gammaERIAL*D.^2);
M1    = Mall(Uindex, Uindex);
M_n   = Mall*S_lab';
M2    = (n_t + 1)/n * M_n;
M_u   = Mall*S_unlab';
M3    = (u_t - 1)/n * M_u;
M2    = M2(Uindex);
M3    = M3(Uindex);
DSVc  = min(D(:, SVindex), [], 2);
fSVc  = exp(DSVc.^2);
sortprob = sort(probs, 2, 'descend');
dBSVSB= sortprob(:,1) - sortprob(:, 2);
C     = fSVc.* dBSVSB;
C     = C(Uindex);
end
function [alpha_opt]   = solveERIALquad(M1, M2, M3, C, beta_ERIAL)
M1 = M1 + 0.001 *eye(numel(C));
alphavar = sdpvar(numel(C), 1);
cConstraint = [ alphavar>=0 , alphavar<=1, sum(alphavar)==1];
cObjective = alphavar'*M1*alphavar + (M2-M3+ beta_ERIAL *C)'*alphavar ;
sol = optimize(cConstraint, cObjective);
if sol.problem == 0
    alpha_opt = value(alphavar);
else
    alpha_opt = -Inf;
end
end