function [kernel_param_list, learning_param_list] = CHR_make_learning_list(learning_params_ref, Classification_exp_param)
% This function makes list of fields list for cross validation for function
% simple-complex
gamma_SCALE_list     = 1;%just svm 0.7:0.1:1.3;
lambda_list    = [10^-1,10^0,10^1];
d_REJ_list        = 0.05:0.05:0.5;
lp_vals        = combvec(d_REJ_list, lambda_list);
n_list         = size(lp_vals,2);
learning_param_list_temp = cell(n_list,1);
n    = 0;
for i= 1:n_list
    learning_params_temp = learning_params_ref;
       learning_params_temp.d_REJ           = lp_vals(1,i);
       learning_params_temp.KOptions.gamma  = lp_vals(2,i)*learning_params_ref.KOptions.gamma;
       learning_params_temp.lambda          = lp_vals(3,i)*learning_params_ref.lambda;
       n = n + 1;
       learning_param_list_temp{n} = set_learningparams(learning_params_temp);
end
learning_param_list      = learning_param_list_temp(1:n);

lp_vals        = combvec(gamma_SCALE_list);
n_list         = size(lp_vals,2);
kernel_param_list_temp = cell(n_list,1);
n    = 0;
for i= 1:n_list
    kernel_param_list_temp = learning_params_ref.KOptions;
    kernel_param_list_temp.gamma  = lp_vals(2,i)*learning_params_ref.KOptions.gamma;
    n = n + 1;
    kernel_param_list_temp{n} = kernel_param_list_temp;
end
kernel_param_list      = kernel_param_list_temp(1:n);
end
