function [learning_param_list] = OUTLIERABLATION_make_learning_list(learning_params_ref, Classification_exp_param)
% This function makes list of fields list for cross validation for function
% simple-complex

lambda_list    = [10^-1,10^0,10^1];
lp_vals        = combvec(lambda_list);
n_list         = size(lp_vals,2);
learning_param_list = cell(n_list,1);
learning_params_temp = learning_params_ref;
for i= 1:n_list
     learning_params_temp.lambda          = lp_vals(1,i)*learning_params_ref.lambda;
     learning_param_list{i}               = learning_params_temp;
end
end
