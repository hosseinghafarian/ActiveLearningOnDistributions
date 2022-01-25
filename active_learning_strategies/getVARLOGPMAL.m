function [exp_par_name, exp_par] = getVARLOGPMAL(experiment_methpar, Almethod_funcs)
exp_par      = experiment_methpar{4};
lambda       = exp_par{1};
lambdastr    = sprintf('%5.3f',lambda);
t_0          = 0;%exp_par{2};
tstr         = sprintf('%3d',t_0);
if numel(exp_par)==3
   mu           = exp_par{3};
else
   mu           = 0;
end
mustr        = sprintf('%2d',mu);
exp_par_name = [getALmethodname(Almethod_funcs),'-t_0:',tstr, '\mu:',mustr, '-\lambda:',lambdastr];
exp_par_name = [getALmethodname(Almethod_funcs)];
end
