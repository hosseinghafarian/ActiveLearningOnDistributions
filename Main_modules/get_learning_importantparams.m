function [lambda, lambda_o, gamma, gamma_o, cp] = get_learning_importantparams(learningparams)
lambda = learningparams.lambda;
lambda_o = learningparams.lambda_o;
gamma    = learningparams.KOptions.gamma;
gamma_o  = learningparams.KOptions.gamma_o;
cp       = learningparams.cp;
end