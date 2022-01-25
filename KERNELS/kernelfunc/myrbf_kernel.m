function [KA, dm, F_ind_row, F_ind_col] = myrbf_kernel(xtrain, Fidxtr, xtest, Fidxte, learningparams, gamma, iseq)
    F_ind_row = xtrain.F;
    F_ind_col = xtest.F;
    if isempty(Fidxtr)
        Fidxtr = true(numel(F_ind_row),1);
    end
    if isempty(Fidxte)
        Fidxte = true(numel(F_ind_col),1);
    end
    dm       = pdist2(xtrain.X(:,Fidxtr)',xtest.X(:,Fidxte)');
    KA       =  comp_Kernel(dm, gamma) ;
end