function [measure_list] = cross_val_kernel_lp_measures(Classification_exp_param, data, myprofile, learning_params_list, kernel_param_list)
global cnstDefs
% split train and cv data into folds
n_param   = numel(learning_params_list);
n_kparam  = numel(kernel_param_list);

p_list_id = 1:n_param;
n_p       = n_param;


lpfork   = cell(n_kparam, 1);

for j=1:n_kparam
    lpfork{j} = learning_params_list{1};
    kernelparam = kernel_param_list{j};
    lpfork{j}.KOptions = kernelparam;
end
lpforkselp = cell(n_p, 1);
for j=1:n_p
    lpforkselp{j} = learning_params_list{p_list_id(j)};
end

data_par= cell(n_kparam, 1);
measure_list = cell(n_kparam, n_p);
mustcompk = ~isfield(Classification_exp_param, 'comp_kernel') || Classification_exp_param.comp_kernel;
parfor j = 1:n_kparam
    if mustcompk
         [data_par{j}] = data.data_comp_kernel(lpfork{j}, Classification_exp_param, data);
    end
    %strmsg = sprintf('Parfor for %d of %d, Please wait...', j, n_kparam); 
    for i = 1:n_p
        learningparam = lpforkselp{i};
        fprintf('\n Cross Validation: iteration:%d of %d', (j-1)*n_p+i, n_p*n_kparam);
        %display_status(mfilename,  4, clmethod, learningparam, data.datasetName);
        % Kernel in data if to be used, must be updated. 
        learningparam.KOptions = kernelparam;
        if mustcompk
           [measure_list{j,i}] = kfold_experiment(Classification_exp_param, @comp_measure, data_par{j}, myprofile, learningparam);
        else
           [measure_list{j,i}] = kfold_experiment(Classification_exp_param, @comp_measure, data, myprofile, learningparam);
        end
        measure_list{j,i}.lp = learningparam;
        measure_list{j,i}.kp = kernelparam; 
    end    
end
end