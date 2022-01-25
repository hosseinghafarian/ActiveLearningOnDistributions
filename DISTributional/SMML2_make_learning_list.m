function [kernel_param_list, learning_param_list] = SMML2_make_learning_list(learning_params_ref, Classification_exp_param)
% This function makes list of fields list for cross validation for function
% SVMDIST
%
load_opt_file = 'save_lp_SMMtrain.mat';
[lp_list] = load_param_from_optimaloffunc(load_opt_file, 10);
n_lpl = numel(lp_list);
for i= 1:n_lpl
   lg_list_opt(i, 1) = lp_list{i}.lambda;
   lg_list_opt(i, 2)  = lp_list{i}.KOptions.gamma;
end
l_opt = unique(lg_list_opt(:,1))';
l_opt_chg = [0.5, 1, 2];
g_opt = unique(lg_list_opt(:,2));
g_opt_chg = [0.7, 1, 1.3];
C_g            = [2^-4, 2^-3, 2^-2, 0.2, 1, 2 ,8 ];%,10];
C_v            = [0.1, 1,  8 ];
Delta          = [ 0.1, 1, 10];
lambda_list    = [2^-4,2^-3,2^-2,2^-1,1];%,2^2];

gamma_SCALE_list     = 0.1:0.05:2;


% C_g            = [2^-4, 2^-3, 2^-2, 0.2, 1, 2 ,4, 8, 16, 32];
% C_v            = [0.1, 1,  8 ];
% Delta          = [ 0.1, 1, 10];
% lambda_list    = [2^-10, 2^-9, 2^-8, 2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,1,2,2^2,2^3, 2^4, 2^5, 2^6, 2^7];
lp_vals        = combvec(l_opt, C_g, C_v, Delta, l_opt_chg);
%lp_vals        = combvec(lambda_list, C_g, C_v, Delta);
n_list         = size(lp_vals,2);
learning_param_list_temp = cell(n_list,1);
n    = 0;
for i= 1:n_list
    learning_params_temp = learning_params_ref;
    learning_params_temp.lambda          = lp_vals(1,i)*lp_vals(5, i);
    learning_params_temp.C_g             = lp_vals(2,i);
    learning_params_temp.C_v             = lp_vals(3,i);
    learning_params_temp.Delta           = lp_vals(4,i);
    n = n + 1;
    learning_param_list_temp{n} = learning_params_temp;
end
learning_param_list      = learning_param_list_temp(1:n);

%lp_vals        = combvec(gamma_SCALE_list);
lp_vals        = combvec(g_opt, g_opt_chg);
n_list         = size(lp_vals,2);
kernel_param_list_temp = cell(n_list,1);
n    = 0;
for i= 1:n_list
    kernel_param_temp = learning_params_ref.KOptions;
    kernel_param_temp.gamma    = lp_vals(1,i)*lp_vals(2,i);
    kernel_param_temp.gamma_is = lp_vals(1,i)*lp_vals(2,i);
    n = n + 1;
    kernel_param_list_temp{n} = kernel_param_temp;
end
kernel_param_list      = kernel_param_list_temp(1:n);
end
