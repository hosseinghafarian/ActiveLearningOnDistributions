function [learning_params] = get_data_learningsettings(data)
    if isfield(data, 'learningparams_init')
        learning_params = data.learningparams_init;
    else%if ~data.isDistData
        [learning_params ]      = learning_settings(data.n, 'gamma'     , data.gamma , 'gamma_is',data.gamma_is, 'gamma_o', data.gamma_o,... 
                                                            'lambda'    , data.lambda, 'lambda_o', data.lambda_o, 'sigma_likelihood', data.sigma_likelihood, ...
                                                            'data_noisy', data.noisy , 'data_labelnoise', data.labelnoise);
    end
end