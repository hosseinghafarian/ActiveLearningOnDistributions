function fname                         = get_exp_filename(Almethod_func, data, learningparams)
    datasetName = data.datasetName;
    gamma    = learningparams.KOptions.gamma;
    gamma_o  = learningparams.KOptions.gamma_o;
    lambda   = learningparams.lambda;
    lambda_o = learningparams.lambda_o;

    expdatafile = strcat('DS=', datasetName); %DS for dataset
    expdatafile = sprintf('%s-%s', expdatafile, '_AL');  %LO for lambda_o
    expdatafile = sprintf('%s=%s', expdatafile, Almethod_func);
    expdatafile = strcat(expdatafile, '_S');    %S for gamma
    expdatafile = sprintf('%s=%4e', expdatafile, gamma);  
    expdatafile = sprintf('%s-%s', expdatafile, '_SO'); %SO for gamma_o
    expdatafile = sprintf('%s=%4e', expdatafile, gamma_o); 
    expdatafile = sprintf('%s-%s', expdatafile, '_LM');  %LM for lambda
    expdatafile = sprintf('%s=%4e', expdatafile, lambda); 
    expdatafile = sprintf('%s-%s', expdatafile, '_LO');  %LO for lambda_o
    expdatafile = sprintf('%s=%4e', expdatafile, lambda_o);
    fname       = strcat(expdatafile, '.mat');
end