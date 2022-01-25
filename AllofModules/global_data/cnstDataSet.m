function cnstDataSet(Kernel,Kernel_o, KA, KA_o, F_to_ind_row, F_to_ind_col, learningparams, al_profile, TrainSamples)
global cnstData
batchSize      = al_profile.batch_size;
maxQuerynumber = al_profile.max_query;
start_notnoisy = al_profile.start_notnoisy;

selectQuerySamples      = TrainSamples.UN_sampling_func;
UN_subsampling_settings = TrainSamples.UN_settings;

selectId  = learningparams.AlselectSamplesMethod;
selectPer = learningparams.AlselectSamplesPercent;
r     = 1.5; 
[u,~,~] = get_nI_diam(TrainSamples.X , r); 
cnstData.Neighbors = u;
%% Set MATRICES Kernel, Kernel matrix between train and test data, Q, etc,
cnstData.K          = Kernel;
cnstData.Lap        = computeLaplacianNormal(Kernel);
cnstData.Q          = eye(size(cnstData.K,1));% or use Q. 
cnstData.Qinv       = inv(cnstData.Q);
mu = 4;
cnstData.KLaplaminv = inv(cnstData.K + learningparams.lambda*cnstData.Q +mu*cnstData.Lap);
cnstData.KLaplaminvexist = true;
dist = pdist2(TrainSamples.X', TrainSamples.X');
[cnstData.rho, cnstData.delta, cnstData.ordrho, ...
    cnstData.nneigh, cnstData.dc, cnstData.ef_rho] = fastcluster_deltarho2(dist, 2);
if ~TrainSamples.isDistData
cnstData.Kclustered = doclusteronk(cnstData.K, cnstData.ef_rho);
cnstData.Kclustered = 1/(1.1)*(cnstData.K +0.1* cnstData.Kclustered);
end
n                   = size(Kernel, 1);
epslamb             = 0.0001;
[cnstData.U,D]      = eig(Kernel+ epslamb *eye(n));
cnstData.D          = diag(D);
cnstData.KA         = KA; %Kernel matrix between train and test data
cnstData.K_o        = Kernel_o;
cnstData.KA_o       = KA_o; %Kernel matrix between train and test data
cnstData.F_to_ind_row= F_to_ind_row;
cnstData.F_to_ind_col= F_to_ind_col;

cnstData.KQinv      = inv(learningparams.rhox*cnstData.Q+learningparams.lambda_o*cnstData.K);
cnstData.Klaminv    = inv(cnstData.K+learningparams.lambda*cnstData.Q);
cnstData.Klamexist  = true;
if ~TrainSamples.isDistData
cnstData.Kclustlaminv = inv(cnstData.Kclustered+learningparams.lambda*cnstData.Q);
cnstData.Kclustlamexist = true;
end
cnstData.H          = learningparams.rhox*cnstData.Q+learningparams.lambda_o*cnstData.K;
cnstData.Hinv       = cnstData.KQinv;

%% Labeled, unlabeled, query sets 
cnstData.batchSize  = batchSize;
%selectquerysamples which is a function variable, is the function with it,
%we determine which instances is chosen for set query. the set query is the
%set of instances from which we select instances to get its label from
%user. 

cnstData.Yl         = TrainSamples.Y;                                   
cnstData.xTrain     = TrainSamples.X;

cnstData.hbar       = 0.7;
%% For Bayesian Learning of Kernel Embedding methods
[cnstData.R, cnstData.BayesK, cnstData.mu_R_alli, cnstData.R_reginv_alli] = computeBayesKernel(learningparams, TrainSamples);

end