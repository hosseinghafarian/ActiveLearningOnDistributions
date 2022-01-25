function cnstDataInitLset(initL, learningparams, al_profile, TrainSamples)
global cnstData
batchSize      = al_profile.batch_size;
maxQuerynumber = al_profile.max_query;
start_notnoisy = al_profile.start_notnoisy;

selectQuerySamples      = TrainSamples.UN_sampling_func;
UN_subsampling_settings = TrainSamples.UN_settings;

selectId  = learningparams.AlselectSamplesMethod;
selectPer = learningparams.AlselectSamplesPercent;

selectQuerySamples      = TrainSamples.UN_sampling_func;
UN_subsampling_settings = TrainSamples.UN_settings;
%% Labeled, unlabeled, query sets 
cnstData.n_S        = TrainSamples.n;
cnstData.initLStart_notnoisy = start_notnoisy;
cnstData.initLStart = initL;
cnstData.initL      = zeros(maxQuerynumber,1);
cnstData.initL(1:numel(initL))      = initL;
cnstData.initLnozero= cnstData.initL>0;
cnstData.n_l        = sum(cnstData.initLnozero);  
cnstData.unlabeled  = setdiff(TrainSamples.F_id, cnstData.initL);
cnstData.n_u        = numel(cnstData.unlabeled);
labeled             = ismember(TrainSamples.F_id, cnstData.initL(cnstData.initLnozero));
%selectquerysamples which is a function variable, is the function with it,
%we determine which instances is chosen for set query. the set query is the
%set of instances from which we select instances to get its label from
%user. 
cnstData.query      = selectQuerySamples(cnstData.K, cnstData.xTrain, cnstData.unlabeled, labeled, cnstData.Yl(labeled), UN_subsampling_settings);%selectQuerySamples(cnstData.unlabeled, selectId, selectPer); % this is the list of instances which we select active learning query from it
cnstData.query      = cnstData.query';
cnstData.n_q        = numel(cnstData.query);
%% Copy query instances to extendInd, Update KE Matrix
cnstData.appendInd  = cnstData.query;                           % at first, append all of selected samples for query to the end of kernel matrix
cnstData.nappend    = numel(cnstData.appendInd);
cnstData.nap        = cnstData.n_S + cnstData.nappend;          % size of alphatild ( alpha + tau) in saddle point optimization
cnstData.extendInd  = cnstData.n_S+1:cnstData.nap;              % appendInd[i]->extendInd[i] 
cnstData.query_to_extend_map = [cnstData.appendInd,cnstData.extendInd']; %to know which instances copied to which extendInd
append_index        = ismember(cnstData.F_to_ind_row, cnstData.appendInd);
Kqq                 = cnstData.K(append_index,append_index);
K_q                 = cnstData.K(:,append_index);                       % Kernel between every thing and setQuery K(labeled and unlabeled,setQuery);
cnstData.KE         = [cnstData.K,K_q;K_q',Kqq];                    % Kernel appended with queryset Kernels with data and with itself
cnstData.L_KE       = norm(cnstData.KE);
cnstData.KEp        = [cnstData.KE,zeros(cnstData.nap,1);zeros(1,cnstData.nap+1)];
cnstData.KEvec      = reshape(cnstData.KEp,numel(cnstData.KEp),1); 
cnstData.Kuvec      = [cnstData.KEvec;zeros(cnstData.n_S,1)];
cnstData.nSDP       = cnstData.nap + 1;          % size of the SDP Matrix 
cnstData.nConic     = cnstData.nSDP*cnstData.nSDP + cnstData.n_S;% this must be increased according to x=(u,...),u=(X,...) 
cnstData.lo         = [zeros(cnstData.n_S,1);-ones(cnstData.nSDP-1-cnstData.n_S,1)];
cnstData.up         = ones(cnstData.nSDP-1,1);   
cnstData.labeled_appendind = intersect(cnstData.appendInd, cnstData.initL);
end