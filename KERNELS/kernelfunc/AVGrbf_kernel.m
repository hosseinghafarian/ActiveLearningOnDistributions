function [KA, dm, F_ind_row, F_ind_col] = AVGrbf_kernel(xtrain, Fidxtr, xtest, Fidxte, learningparams, gamma, iseq)
    F_ind_row = xtrain.F_id;
    F_ind_col = xtest.F_id;
    if isempty(Fidxtr)
        Fidxtr = true(numel(F_ind_row),1);
    end
    if isempty(Fidxte)
        Fidxte = true(numel(F_ind_col),1);
    end
    indtr    = ismember(F_ind_row, Fidxtr);
    indte    = ismember(F_ind_col, Fidxte);
    dm       = pdist2(xtrain.X_avg(:,indtr)',xtest.X_avg(:,indte)');
    KA       =  comp_Kernel(dm, gamma) ;
end