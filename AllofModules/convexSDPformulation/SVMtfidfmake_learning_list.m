function [kernel_param_list, learning_param_list] = SVMtfidfmake_learning_list(learning_params_ref, Classification_exp_param)
% This function makes list of fields list for cross validation for function
% simple-complex
%c_list         = 0.049:0.05:0.499;
gamma_SCALE_list     = 0.5:0.1:1.5;
lambda_list    = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,1,2,2^2,2^3];

lp_vals        = combvec(lambda_list);
n_list         = size(lp_vals,2);
learning_param_list_temp = cell(n_list,1);
n    = 0;
for i= 1:n_list
    learning_params_temp = learning_params_ref;
    learning_params_temp.lambda          = lp_vals(1,i)*learning_params_ref.lambda;
    n = n + 1;
    learning_param_list_temp{n} = set_learningparams(learning_params_temp);
end
learning_param_list      = learning_param_list_temp(1:n);

lp_vals        = combvec(gamma_SCALE_list);
n_list         = size(lp_vals,2);
kernel_param_list_temp = cell(n_list,1);
n    = 0;
for i= 1:n_list
    kernel_param_temp = learning_params_ref.KOptions;
    kernel_param_temp.gamma  = lp_vals(1,i)*learning_params_ref.KOptions.gamma;
    n = n + 1;
    kernel_param_list_temp{n} = kernel_param_temp;
end
kernel_param_list      = kernel_param_list_temp(1:n);
end
