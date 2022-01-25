function [exp_par_name, exp_par] = getVARLOGPMAL(experiment_methpar, Almethod_funcs)
exp_par      = experiment_methpar{4};
lambda       = exp_par{1};
lambdastr    = sprintf('%5.3f',lambda);
if numel(exp_par)==3
   t_0          = exp_par{2};
   tstr         = sprintf('%3d',t_0);
   mu           = exp_par{3};
   mustr        = sprintf('%2d',mu);
   str = ['-t_0:',tstr, '\mu:',mustr, '-\lambda:',lambdastr]
elseif numel(exp_par)==1
    
   str = [':',lambdastr];
end

exp_par_name = [getALmethodname(Almethod_funcs),str];
%exp_par_name = [getALmethodname(Almethod_funcs)];
end
