function [KA, dm, F_ind_row, F_ind_col] = rbf_twolevel(xtrain, Fidxtr, xtest, Fidxte, learningparams, gamma, iseq)
    [dm, F_ind_row, F_ind_col] = distance_matrix(xtrain, Fidxtr, xtest, Fidxte, learningparams.KOptions.gamma_is, iseq);
    KA    = comp_Kernel(dm, gamma);
end