function [f_test_val] = SMM3waytest(model, learningparams, data, idx)
% distu       = unique(data.F); % what are the unique distnums 
% distidx     = distu(idx);     % which unique distnums are for training
% testi       = ismember(data.F, distidx); % select instaces which belongs to distributions in idx
% data_x_test = data.X(:,testi);
% F           = data.F(testi);
% testFidx    = ismember(unique(F), data.F);
% [KA]        = DISTKernelArray(model, data_x_test, F, learningparams);
global cnstDefs
idxF_K      = ismember(data.F_to_ind_row, model.uF_K);
KA          = data.K_3way(idxF_K, idx);
n_test      = sum(idx);

if model.use_libsvm 
    cmdstr = ' -b 0 ';
%     if ~cnstDefs.solver_verbose 
%         cmdstr = strcat(cmdstr,' -q ');
%     end
    KA_indexed = [(1:n_test)',KA'];
    [predict_label, accuracy_seq, f_test_val] = svmpredict((1:n_test)', KA_indexed, model.libsvmmodel, cmdstr);
    accuracy  = accuracy_seq(1);
else
    f_test_val  = KA'*model.w + model.b_w*ones(n_test,1);
end
end