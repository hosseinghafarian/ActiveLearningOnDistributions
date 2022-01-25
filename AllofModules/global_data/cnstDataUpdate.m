function [initLcurrnexInd] = cnstDataUpdate(queryInstance)
% This function change query and unlabeled instances and updates KE matrices after addition of 
% queryInstance
global cnstData
% pre_? is used for warmstarting using previous values of x 
cnstData.pre_nSDP = cnstData.nSDP;
cnstData.pre_nappend = cnstData.nappend;
cnstData.pre_query_to_extend_map = cnstData.query_to_extend_map;
% update initL
[cnstData.initL, initLcurrnexInd] = append_new_labeled_instance( cnstData.initL, queryInstance, cnstData.batchSize);
cnstData.initLnozero= cnstData.initL>0;
cnstData.n_l        = sum(cnstData.initLnozero);
% update unlabeled 
cnstData.unlabeled  = setdiff(cnstData.unlabeled, cnstData.initL);
cnstData.n_u        = numel(cnstData.unlabeled);
% update query 
cnstData.query      = setdiff(cnstData.query, queryInstance);
cnstData.n_q        = numel(cnstData.query);
% update extendind, KE matrix and other depenedent quantities
cnstData.appendInd  = cnstData.query;                           % at first, append all of selected samples for query to the end of kernel matrix
cnstData.nappend    = numel(cnstData.appendInd);
cnstData.nap        = cnstData.n_S + cnstData.nappend;          % size of alhatild ( alpha + tau) in saddle point optimization
cnstData.extendInd  = cnstData.n_S+1:cnstData.nap;              % appendInd[i]->extendInd[i] 
cnstData.query_to_extend_map = [cnstData.appendInd, cnstData.extendInd']; %to know which instances copied to which extendInd
append_index        = ismember(cnstData.F_to_ind_row, cnstData.appendInd);
Kqq                 = cnstData.K(append_index,append_index);
K_q                 = cnstData.K(:,append_index);                      % Kernel between every thing and setQuery K(labeled and unlabeled,setQuery);
compute_KE = false; 
if compute_KE 
cnstData.KE         = [cnstData.K,K_q;K_q',Kqq];                    % Kernel appended with queryset Kernels with data and with itself
cnstData.L_KE       = norm(cnstData.KE);
cnstData.KEp        = [cnstData.KE,zeros(cnstData.nap,1);zeros(1,cnstData.nap+1)];
cnstData.KEvec      = reshape(cnstData.KEp,numel(cnstData.KEp),1); 
cnstData.Kuvec      = [cnstData.KEvec;zeros(cnstData.n_S,1)];
end
% update size of the matrices 
cnstData.nSDP       = cnstData.nap + 1;          % size of the SDP Matrix 
cnstData.nConic     = cnstData.nSDP*cnstData.nSDP + cnstData.n_S;% this must be increased according to x=(u,...),u=(X,...) 
cnstData.lo         = [zeros(cnstData.n_S,1);-ones(cnstData.nSDP-1-cnstData.n_S,1)];
cnstData.up         = ones(cnstData.nSDP-1,1);   
cnstData.labeled_appendind = intersect(cnstData.appendInd, cnstData.initL);
end
function [initL, initLcurrnexInd] = append_new_labeled_instance( initL, queryInstance, batchSize)
initLnozero= initL>0;
initLcurrnexInd  = sum(initLnozero)+1;
assert(sum(ismember(queryInstance,initL))==0,'Cannot query an instance twice');
initL(initLcurrnexInd:initLcurrnexInd+batchSize-1) = queryInstance; 
initLcurrnexInd  = initLcurrnexInd + batchSize;
end