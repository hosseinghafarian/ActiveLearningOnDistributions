function [learning_param_list] = RELSV4CL_make_learning_list(learning_params_ref, Classification_exp_param)
% This function makes list of fields list for cross validation for function
% simple-complex
gamma_list     = 1;%[1,10,100];
lambda_list    = [10^-1,10^0,10^1];
noiserate_list = 0:0.05,0.5;
lp_vals        = combvec(gamma_list, lambda_list, noiserate_list);
n_list         = size(lp_vals,2);
learning_param_list = cell(n_list,1);
learning_params_temp = learning_params_ref;
for i= 1:n_list
     learning_params_temp.KOptions.gamma  = lp_vals(1,i)*learning_params_ref.KOptions.gamma;
     learning_params_temp.lambda          = lp_vals(2,i)*learning_params_ref.lambda;
     learning_params_temp.noiserate       = lp_vals(3,i);
     learning_param_list{i}               = learning_params_temp;
end
end
