function [exp_par_name, alparams] = getSPPMALEPstr(experiment_methpar, Almethod_funcs)
alparams      = experiment_methpar{4};

lambda  = alparams{1};
mu      = alparams{2};

tstr      = sprintf('-\lambda:%5.2f-\mu:%5.2f',lambda, mu);
exp_par_name = [getALmethodname(Almethod_funcs), tstr];
end
