function [learning_params, measures] = myfastcrossval(Classification_exp_param, data, myprofile, learning_params_init)
global cnstDefs
top_p_per = 0.5;
mycrossvalprofile           = myprofile;

learning_list_maker  = Classification_exp_param.making_learning_list_func;
clmethod             = Classification_exp_param.clmethod;
display_status(mfilename,  3, clmethod );
[kernel_param_list, learning_param_list]= learning_list_maker(learning_params_init, Classification_exp_param);
myprofile.CV_search_percent = 30;
myprofile.Full_CV_search      = false;
[learning_params_list]            = select_subset(learning_param_list, myprofile);
% select train and cv data
data.DS_settings.percent = 100;
[data ]   = data.DS_subsampling_func(data,  data.DS_settings);
% split train and cv data into folds
n_param   = numel(learning_params_list);
n_kparam  = numel(kernel_param_list);
p_list_id = 1:n_param;
n_p       = n_param;

lp_sel    = true(n_param, 1);
kp_sel    = true(n_kparam, 1);
lpc       = learning_params_list;
kpc       = kernel_param_list;
n_full    = 480;
n_ml      = n_param*n_kparam;
endcv     = false;
h = waitbar(0, 'Please wait');
for crdataratio = 0.05:0.1:0.65
    if n_ml > n_full
       mycrossvalprofile.trainratio= crdataratio;
       mycrossvalprofile.testratio = 1-crdataratio;
       [data_cr, ~] =  selectTrainAndTestSamples(data , mycrossvalprofile, data.TR_sampling_func, data.TR_settings);
    else
       data_cr  = data; 
       endcv = true;
    end
    tic;
    [measure_list] = cross_val_kernel_lp_measures(Classification_exp_param, data_cr, myprofile, lpc, kpc);
    toc
    if endcv
        break;
    end
    n_ml  = numel(measure_list);
    [lp_sel, kp_sel] = select_top_p_per(measure_list, top_p_per);    
    lpc = lpc(lp_sel);
    kpc = kpc(kp_sel);
    n_ml = numel(lpc)*numel(kpc);
    waitbar(crdataratio/0.8, h);
end    
close(h);
[n_k, n_p] = size(measure_list);
measure_listsq = cell(n_k*n_p, 1);

for j= 1:n_k
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
function [lp_sel, kp_sel] = select_top_p_per(measure_list, p)
[n_k, n_p] = size(measure_list);
lp_sel     = false(n_p,1);
kp_sel     = false(n_k,1);
top_n   = (floor(p*n_p*n_k) + 1);
acc = zeros(n_k, n_p);
for i = 1:n_k
    for j= 1:n_p
        acc(i, j) = measure_list{i,j}.acc_avg;
    end
end
acrsum  = sum(acc, 1);
accsum  = sum(acc, 2);
[~, indcr] = sort(acrsum, 'descend');
[~, indcc] = sort(accsum, 'descend');
top_nr = floor(n_p*p) + 1;
for i = 1:top_nr
   lp_i = indcr(i);
   lp_sel(lp_i) = true;
end
top_nr = floor(n_k*p) + 1;
for i = 1:top_nr
   
   kp_i = indcc(i);

   kp_sel(kp_i) = true;
end
% acind       = combvec(1:n_k, 1:n_p);
% indlin      = sub2ind(size(acc), acind(1,:), acind(2,:));
% [~, indtop] = sort(acc(indlin),  'descend');   
% [actopind_r, actopind_c]    = ind2sub(size(acc), indtop);
% 
% for i = 1:top_n
%    lp_i = actopind_c(i);
%    kp_i = actopind_r(i);
%    lp_sel(lp_i) = true;
%    kp_sel(kp_i) = true;
% end
end