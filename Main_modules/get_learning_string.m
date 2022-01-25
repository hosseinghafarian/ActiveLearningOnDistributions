function [str] = get_learning_string(learningparams)
[lambda, lambda_o, gamma, gamma_o, cp] = get_learning_importantparams(learningparams);
str   = sprintf('\\gamma=%7.3e  \\gamma_o=%7.3e \\lambda=%7.3e \\lambda_o=%7.3e C_p=%5.3f',gamma, gamma_o, lambda, lambda_o, cp);
end