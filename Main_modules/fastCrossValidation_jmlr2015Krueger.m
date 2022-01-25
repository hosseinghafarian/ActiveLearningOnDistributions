function [learningparams, measures] = fastCrossValidation_jmlr2015Krueger(Classification_exp_param, data, myprofile, learning_params_init)
global cnstDefs
learning_list_maker  = Classification_exp_param.making_learning_list_func;
clmethod             = Classification_exp_param.clmethod;
display_status(mfilename,  3, clmethod );
data.DS_settings.percent = 100;
[data ]   = data.DS_subsampling_func(data,  data.DS_settings);

[kernel_param_list, learning_param_list]= learning_list_maker(learning_params_init, Classification_exp_param);
kp_ind  = 1:numel(kernel_param_list);
lp_ind  = 1:numel(learning_param_list);
[kplp_ind] = combvec(lp_ind, kp_ind)';
N       = data.n;
ordind  = randperm(N);
C       = size(kplp_ind, 1);
S       = 20;  % number of steps of cross validation
Delta   = floor(N/S) + 1;
beta_l  = 0.2;
alpha_l = 0.01;
alphafast = 0.01;
w_stop  = 10;

n_f     = Delta;
T_s     = zeros(C, S);
P_s     = Inf(C, S);
isActive= true(C, 1);
h = waitbar(0, 'Please wait ... cross validation');
s = 1;
while (s <=S && (n_f + 1) < N)
   P_p = Inf(C, N-n_f); % Pointwise performance matrix P_p stores the TEST ERRORS 

   Trainnumind       = sort(ordind(1:n_f));
   Testnumind        = sort(ordind(n_f+1:N));
   [TrainSamples, TestSamples] = data.split_data_func(data, Trainnumind, Testnumind);
   prekp_ind = -1;
   for c = 1:C
      if isActive(c)
          kp_ind  = kplp_ind(c, 2);
          learningparams          = learning_param_list{kplp_ind(c, 1)};
          learningparams.KOptions = kernel_param_list{kp_ind};
          if kp_ind ~= prekp_ind
              recompute_kernel = true;
              prekp_ind = kp_ind;
          else
              recompute_kernel = false;
          end
          if recompute_kernel &&( ~isfield(Classification_exp_param, 'comp_kernel') || Classification_exp_param.comp_kernel)
              fprintf('Recomputing Kernel:iteration %d, config %d of %d\n', s, c, C);
             [TrainSamples] = TrainSamples.data_comp_kernel(learningparams, Classification_exp_param, TrainSamples);
             [TestSamples]  = TestSamples.data_comp_kernel(learningparams, Classification_exp_param, TestSamples, TrainSamples);
          end
          try 
             [P_p_c, P_p_avg] = learnonfirst_evalonrest(Classification_exp_param, myprofile, learningparams, ...
                                                     TrainSamples, TestSamples);
          catch ME
             P_p_c = 0;
             P_p_avg = 0;
          end
          P_p(c,:) = P_p_c';
          P_s(c,s) = P_p_avg;
      end
      waitbar(((s-1)*C+c)/(C*S),h);
   end
   K        = sum(isActive);
   if K==1
       break;
   end
   indextop = TopConfiguration(P_p, alphafast, K);
   T_s(indextop, s) = 1;
   for c = 1:C
      if isActive(c)&& is_flopConfiguration(T_s(c, 1:s), s, S, beta_l, alpha_l)
          isActive(c) = false;
%           [deAc_c] = deactive_similar(isActive, kplp_ind, c, s, S);
%           isActive(deAc_c) = false;
      end
   end
   if (s>=w_stop)&& similarPerformance(T_s(isActive, max(s-w_stop, 0)+1:s), alphafast) 
       break;
   end    
   n_f = n_f + Delta;
   s   = s + 1;
end 
close(h);
[M_avg, Ind] = selectWinner(P_s, isActive, w_stop, s, C);
kp_ind  = kplp_ind(Ind, 2);
learningparams          = learning_param_list{kplp_ind(Ind, 1)};
learningparams.KOptions = kernel_param_list{kp_ind};

measures     = M_avg; 
end
function [deAc_c] = deactive_similar(isActive, kplp_ind, c, s, S)
similarkp = kplp_ind(c,2);
issimilar = kplp_ind(:, 2)==similarkp;
deacratio = sum(isActive(issimilar))/sum(issimilar);
if deacratio <= 1-s/S
   deAc_c = issimilar;
else
   deAc_c = [];
end
end
function [M, Ind] = selectWinner(Ps,  isActive, w_stop, s, C)
   R_S = Inf(C, s);
   for i=1:s
       for c= 1:C
           if isActive(c)
               R_S(c, i) = sum(Ps(c,i) <= Ps(:,i));
           end
       end
       M_S = Inf(C, 1);
       for c = 1:C
           if isActive(c)
               M_S(c) = (1/w_stop)*sum(R_S(c, s-w_stop+1:s));
           end
       end
   end    
   [M, Ind] = min(M_S);
end


% for j = 1:n_kparam
%     kernelparam = kernel_param_list{j};
%     lpfork.KOptions = kernelparam;
%     if ~isfield(Classification_exp_param, 'comp_kernel') || Classification_exp_param.comp_kernel
%          [data] = data.data_comp_kernel(lpfork, Classification_exp_param, data);
%     end
%     for i = 1:n_p
%         iparam = p_list_id(i);
%         learningparam = learning_params_list{iparam};
%         disp(sprintf('Cross Validation: iteration:%d of %d', t, n_p*n_kparam));
%         display_status(mfilename,  4, clmethod, learningparam, data.datasetName);
%         % Kernel in data if to be used, must be updated. 
%         learningparam.KOptions = kernelparam;
%         
%         [measure_list{t}] = kfold_experiment(Classification_exp_param, @comp_measure, data, myprofile, learningparam);
%         measure_list{t}.lp = learningparam;
%         measure_list{t}.kp = kernelparam;
%         t = t + 1;
%     end
% end
% [measures]        = array_of_struct_to_struct_of_array(measure_list);
% measures.measure_list = measure_list;
% fn = get_savefilename(true, cnstDefs.result_classification_path , data.datasetName, 'CROSSVALIDATION', clmethod );
% save(fn, 'measures','learning_params_list');
% [lp, kp] = select_max_measures(measures);
% learning_params = lp;
% learning_params.KOptions = kp;
% end
% function [lp, kp] = select_max_measures(measures)
% [val, imax_acc]   = max(measures.acc_avg);
% % id_max_plist      = p_list_id(imax_acc);
% % learning_params   = learning_params_list{id_max_plist}; 
% lp = measures.measure_list{imax_acc}.lp;
% kp = measures.measure_list{imax_acc}.kp;
% 
% end