function [f_test_val] = Simple_complex_test_1(model, learningparams, data, idx)
data_x_test = data.X(:,idx);

[KA, KA_o]     = get_two_KernelArray(model.trainx, data_x_test, learningparams, true);
f_test_val     = KA'*(model.alpha.*model.h.*model.trainy');

end