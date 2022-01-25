function [R, BayesK, mu_R_alli, R_reginv_alli] = computeBayesKernel(learningparams, data)

gamma   = learningparams.KOptions.gamma_is;
gamma_r = learningparams.KOptions.gamma_r;
thau    = learningparams.BME_thau; 
%[R2, KB2, Rreginv_cells, mu_R_cells] = compute_KBayes_m( data.X, data.F, gamma, gamma_r, thau);
[R, BayesK, mu_R_alli, R_reginv_alli] = compute_KBayes(data.X, data.F, gamma, gamma_r, thau);
end