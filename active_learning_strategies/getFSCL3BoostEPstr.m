function [exp_par_name, alparams] = getFSCL3BoostEPstr(experiment_methpar, Almethod_funcs)
alparams      = experiment_methpar{4};

trho  = alparams{1};
thau = alparams{2};
t_step    = alparams{3};

top_n     = alparams{4};
tstr      = sprintf('-\\rho:%3d-\\thau:%2d:%2d-m:%2d',trho, thau, t_step, top_n);
exp_par_name = [getALmethodname(Almethod_funcs), tstr];
end
