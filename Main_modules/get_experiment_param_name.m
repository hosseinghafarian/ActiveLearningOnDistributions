function [exp_par_names, exp_pars] = get_experiment_param_name(almethod_list, active_experiment_sequence)
n_e = numel(active_experiment_sequence);
exp_par_names = cell(n_e, 1);
exp_pars      = cell(n_e, 1);
for i=1:n_e
   experiment_methpar         = active_experiment_sequence{i};
   [Almethod_funcs, j, found] = get_method_funcs(almethod_list, experiment_methpar{1});
   getEPstr        = Almethod_funcs{8};
   [exp_par_names{i}, exp_pars{i}] = getEPstr(experiment_methpar, Almethod_funcs); 
end
end

