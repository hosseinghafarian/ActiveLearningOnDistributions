function [objective] = primal_regwo_objective(x_k, Ghat, operators, learningparams, optparams)
global cnstData
    objective = primal_objective(x_k, Ghat)+ learningparams.lambda_o/2*x_k.w_obeta'*cnstData.K*x_k.w_obeta;
end