function [exp_par_name, alparams] = getFSCL2BoostEPstr(experiment_methpar, Almethod_funcs)
alparams      = experiment_methpar{4};

t_simple  = alparams{1};
t_complex = alparams{2};
t_step    = alparams{3};

top_n     = alparams{4};
tstr      = sprintf('-t:%3d:%2d:%3d-b:%2d',t_simple, t_complex, t_step, top_n);
exp_par_name = [getALmethodname(Almethod_funcs), tstr];
end
