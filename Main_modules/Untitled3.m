function [fname] = getfilename_basedon_learningimportant_params(learningparams)
global cnstDefs
[lambda, lambda_o, gamma, gamma_o, cp] = get_learning_importantparams(learningparams);
path  = cnstDefs.Results;
str   = sprintf('_S=%7.5f_SO=%7.5f_L=%7.5f_LO=%7.5f_CP=%5.3f',lambda, lambda_o, gamma, gamma_o, cp);
fname = [path,'\',str];

end