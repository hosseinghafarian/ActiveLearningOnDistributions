function [f_test_val] = Complex_test_1(model, learningparams, data_x_test)

[KA, KA_o]     = get_two_KernelArray(model.trainx, data_x_test, learningparams, true);
f_test_val     = KA_o'*(model.w_obeta.*model.trainy');

end