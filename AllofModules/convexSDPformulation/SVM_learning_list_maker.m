function [learning_param_list] = SVM_learning_list_maker(learning_params_ref, Classification_exp_param)
% This function makes list of fields list for cross validation for function
% simple-complex
%c_list         = 0.049:0.05:0.499;
gamma_SCALE_list = 0.5:0.5:10;
lambda_list      = 1;%[10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2,10^3,10^4,10^5];
lp_vals        = combvec(gamma_SCALE_list, lambda_list);
n_list         = size(lp_vals,2);
learning_param_list_temp = cell(n_list,1);
for i= 1:n_list
    learning_params_temp                 = learning_params_ref;
    learning_params_temp.KOptions.gamma  = lp_vals(1,i)*learning_params_temp.KOptions.gamma;
    learning_params_temp.lambda          = lp_vals(2,i);
    learning_param_list_temp{i}          = set_learningparams(learning_params_temp);
end
learning_param_list      = learning_param_list_temp(1:n_list);
end
