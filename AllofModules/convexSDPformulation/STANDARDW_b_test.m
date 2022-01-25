function [f_test_val] = STANDARDW_b_test(model, learningparams, data, idx)
data_x_test = data.X(:,idx);
[KA, KA_o]     = get_two_KernelArray(model.trainx, data_x_test, learningparams, true);
n_test         = size(data_x_test,2);
f_test_val     = KA'*model.w + model.b_w*ones(n_test,1);

end