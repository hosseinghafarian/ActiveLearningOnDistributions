function [exp_par_name, exp_par] = getquireEPstr(experiment_methpar, Almethod_funcs)
exp_par      = experiment_methpar{4};
lambda       = exp_par{1};
lambdastr    = sprintf('%5.3f',lambda);

exp_par_name = [getALmethodname(Almethod_funcs),'-\lambda:',lambdastr];
end
