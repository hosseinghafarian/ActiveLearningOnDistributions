function [exp_par_name, exp_par] = getDUMMYEPstr(experiment_methpar, Almethod_funcs)
exp_par      = experiment_methpar{4};
exp_par_name = getALmethodname(Almethod_funcs);
end
