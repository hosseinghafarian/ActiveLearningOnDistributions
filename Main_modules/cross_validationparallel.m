function [learning_params, measures] = cross_validationparallel(Classification_exp_param, data, myprofile, learning_params_init)
global cnstDefs
learning_list_maker  = Classification_exp_param.making_learning_list_func;
clmethod             = Classification_exp_param.clmethod;
display_status(mfilename,  3, clmethod );
[kernel_param_list, learning_param_list]= learning_list_maker(learning_params_init, Classification_exp_param);
[learning_params_list]            = select_subset(learning_param_list, myprofile);
% select train and cv data
data.DS_settings.percent = 100;
[data ]   = data.DS_subsampling_func(data,  data.DS_settings);
% split train and cv data into folds
n_param   = numel(learning_params_list);
n_kparam  = numel(kernel_param_list);
p_list_id = 1:n_param;
n_p       = n_param;
measure_list       = cell(n_p,1);
t = 1;
lpfork   = cell(n_kparam, 1);
data_par = cell(n_kparam, 1);
for j=1:n_kparam
    lpfork{j} = learning_params_list{1};
    kernelparam = kernel_param_list{j};
    lpfork{j}.KOptions = kernelparam;
    %data_par{j} = data;
end
lpforkselp = cell(n_p, 1);
for j=1:n_p
    lpforkselp{j} = learning_params_list{p_list_id(j)};
end    
data_par= cell(n_kparam, 1);
measure_list = cell(n_kparam, n_p);
mustcompk = ~isfield(Classification_exp_param, 'comp_kernel') || Classification_exp_param.comp_kernel;
fprintf('Running classifier %d times %d times %d = %d times', n_kparam, n_p, 10, n_kparam*n_p*10);
htt = tic;

parfor j = 1:n_kparam
    tic;
    if mustcompk
         [data_par{j}] = data.data_comp_kernel(lpfork{j}, Classification_exp_param, data);
    end
    toc
    strmsg = sprintf('Parfor for %d of %d, Please wait...', j, n_kparam); 
%     h(j)  = waitbar(0,strmsg);
    for i = 1:n_p
        learningparam = lpforkselp{i};
        fprintf('Cross Validation: iteration:%d of %d', (j-1)*n_p+i, n_p*n_kparam);
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
%         waitbar(i/n_p, h(j));
    end    
%     close(h(j));
end
toc(htt)
measure_listsq = cell(n_kparam*n_p, 1);
for j= 1:n_kparam
    for i=1:n_p
        measure_listsq{(j-1)*n_p+i} =  measure_list{j,i};
    end
end
try 
   [measures]        = array_of_struct_to_struct_of_array(measure_listsq);
catch
    disp('oh');
end
measures.measure_list = measure_list;
% fn = get_savefilename(true, cnstDefs.result_classification_path , data.datasetName, 'CROSSVALIDATION', clmethod );
% save(fn, 'measures','learning_params_list');
[lp, kp] = select_max_measures(measures);
learning_params = lp;
learning_params.KOptions = kp;
end
function [lp, kp] = select_max_measures(measures)
[val, imax_acc]   = max(measures.acc_avg);
% id_max_plist      = p_list_id(imax_acc);
% learning_params   = learning_params_list{id_max_plist}; 
lp = measures.measure_list{imax_acc}.lp;
kp = measures.measure_list{imax_acc}.kp;

end
% show_mesh(learning_params_list, learning_param_fields, AVG_param_results,'Accuracy',...
%               [learning_param_fields.firstfield,'_accuracy_',data.datasetName]);
% show_mesh(learning_params_list, learning_param_fields, AVG_param_noiseacc,'Noise Recognition Rate',...
%               [learning_param_fields.secondfield,'_noise_recog_rate_',data.datasetName]);
% show_mesh(learning_params_list, learning_param_fields, AVG_p_values(:,1),'Maximum Value of P for noise',...
%               [learning_param_fields.secondfield,'_max_p_noise',data.datasetName]);
% show_mesh(learning_params_list, learning_param_fields, AVG_p_values(:,2),'Average Value of P for noise',...
%               [learning_param_fields.secondfield,'_avg_p_noise',data.datasetName]);
% show_mesh(learning_params_list, learning_param_fields, AVG_p_values(:,3),'Average Value of P for non-noise',...
%               [learning_param_fields.secondfield,'_avg_p_non_noise',data.datasetName]);          
%return learning_params_list{imax results from cv_res}