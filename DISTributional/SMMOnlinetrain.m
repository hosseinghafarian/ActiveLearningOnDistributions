function [model] = SMMOnlinetrain(learningparams, data, idx )
global cnstDefs

    uselibsvm    = true;
    data_train_y = data.Y(idx);
    n            = numel(data_train_y);     % size of data 
    
%   distu        = unique(data.F); % what are the unique distnums 
    distidx      = data.F_id(idx);     % which unique distnums are for training
    traini       = ismember(data.F, distidx); % select instaces which belongs to distributions in idx
    vecdata_train_x = data.X(:, traini);
    id_list      = distidx;
    
options.eta    =  0.2;
options.lambda =  1/(n^2);
options.gamma  =  10;
options.t_tick = round(n/15);
options.sigma  = 8;
%options for bpas
options.C_BPAs = 1;
%options for NOGD
options.eta_nogd=0.2;
options.k = 0.2;
%options for FOGD
options.eta_fou=0.002;
options.D=4;
B=[50 100 200 300 400 500];
options.Budget = B(2);
data.learningparams = learningparams;

%[classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = NysGDSMMOnline(data, options, id_list); 
[classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = SMMOGD(data, options, id_list); 

idx_SV = classifier.SV;
[data] = data.data_comp_kernel(data.learningparams, data.Classification_exp_param, data, data, idx_SV, idx_SV);
K_SV_SV = data.K;
idx      = ismember(data.F_id, idx_SV);

[model] = SMMtrain(learningparams, data, idx , K_SV_SV);
model.SV = idx_SV;
model.SMMOnline = true;
end