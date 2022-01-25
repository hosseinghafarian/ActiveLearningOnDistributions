function [f_test_val] = STANDARDW_test(model, learningparams, data_x_test)

[KA, KA_o]     = get_two_KernelArray(model.trainx, data_x_test, learningparams, true);
f_test_val     = KA'*model.w ;
end