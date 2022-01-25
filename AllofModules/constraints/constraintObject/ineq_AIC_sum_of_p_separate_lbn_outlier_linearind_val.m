function [Rind, Rvalue, s, constraint_to_instance_map, y_new, x_st_new, size_of_s,nzmax]   = ineq_AIC_sum_of_p_separate_lbn_outlier_linearind_val(pre_constr_inst_map_IC, pre_y,pre_x_st, c_mul_pAndw_o,elements_of_diag)
    % cConstraint=[cConstraint,sum(p(all-initL)<=n_u*onoiseper/100]; %sum(p(initL))<=n_l*lnoiseper/100,
    % Constraint: 1^T p(initL) <= n_l*lnoiseper/100, 1^T p(all-initL) <= n_u*onoiseper/100
global cnstData 
global cnstDefs

    nSDP      = cnstData.nSDP;
    n_S       = cnstData.n_S;
    initL     = cnstData.initL(cnstData.initL>0);
    initLStart          = cnstData.initLStart; 
    initLStart_notnoisy = cnstData.initLStart_notnoisy;
    if initLStart_notnoisy 
       initialinst       = setdiff(initL, initLStart);
       init_start        = initLStart;
    else
       initialinst       = initL;
       init_start        = [];
    end
    %% For initLStart: not noisy
%     n_ls      = numel(init_start);
%     n_nls     = n_S - n_ls;
%     nzmax     = max(n_ls, n_nls);
%     rows      = [];
%     cols      = [];
%     linear_row= 1;
%     zeronl    = ones(1,n_ls);
%     onenl     = ones(1,n_ls);
%     plinear_row = onenl;
%     vals      = 0;
%     if ~isempty(pre_constr_inst_map_IC)
%        ind             = pre_constr_inst_map_IC(:,3)==cnstDefs.CSTR_ID_NOISE_P_INITL & ismember(pre_constr_inst_map_IC(:,2),initLStart);
%        y_new_pinitls    = pre_y(ind);
%        x_st_new_pinitls = pre_x_st(ind);
%     else
%        y_new_pinitls    = 0; 
%        x_st_new_pinitls = 0;
%     end
%     if n_ls==0, s_start = 0; else s_start = 1; end
%     constraint_to_instance_map_initLs = [ cnstDefs.LABELED_INS, 0, cnstDefs.CSTR_ID_NOISE_P_INITL];
%     [Rind_lbns , Rvalue_lbns] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_row, ...
%                                 rows, cols, vals, plinear_row, initialinst, zeronl);
    %% For initL but may be noisy
    n_l       = numel(initialinst);
    n_nl      = numel(cnstData.unlabeled);
    nzmax     = max(n_l, n_nl);
    rows      = [];
    cols      = [];
    linear_row= 1;
    onenl     = ones(1,n_l);
    plinear_row = onenl;
    vals      = 0;
    if ~isempty(pre_constr_inst_map_IC)
       ind = pre_constr_inst_map_IC(:,3)==cnstDefs.CSTR_ID_NOISE_P_INITL & ismember(pre_constr_inst_map_IC(:,2),initialinst);
       y_new_pinitl = pre_y(ind);
       x_st_new_pinitl = pre_x_st(ind);
    else
       y_new_pinitl    = 0; 
       x_st_new_pinitl = 0;
    end
    if n_l==0
       size_of_s_labeled_notstart = 0;
       constraint_to_instance_map_initL = [];
       y_new_pinitl    = []; 
       x_st_new_pinitl = [];
       s_labeled_notstart = [];
    else
       constraint_to_instance_map_initL = [ cnstDefs.LABELED_INS, 0, cnstDefs.CSTR_ID_NOISE_P_INITL]; 
       size_of_s_labeled_notstart = 1; 
       s_labeled_notstart = cnstData.n_l*cnstData.lnoiseper;
    end
    
    [Rind_lbn , Rvalue_lbn] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_row, ...
                                rows, cols, vals, plinear_row, initialinst, onenl);        
    %% For unlabeled
    unlab     = cnstData.unlabeled;    
    onen_un   = ones(1,n_nl);
    plinear_row = onen_un;
    vals      = 0;
    if ~isempty(pre_constr_inst_map_IC)
       ind = pre_constr_inst_map_IC(:,3)==cnstDefs.CSTR_ID_NOISE_P_UNLABELED;
       y_new_punlabeled = pre_y(ind);
       x_st_new_punlab  = pre_x_st(ind);
    else
       y_new_punlabeled = 0; 
       x_st_new_punlab  = 0;
    end
    if n_nl==0
       size_of_s_unlab = 0; 
       constraint_to_instance_map_unlab = [];
       s_unlab         = [];
    else
       size_of_s_unlab = 1; 
       constraint_to_instance_map_unlab = [ cnstDefs.UNLABELED_INS, 0, cnstDefs.CSTR_ID_NOISE_P_UNLABELED];
       s_unlab         = cnstData.n_u*cnstData.onoiseper;
    end;
    [Rind_outlier , Rvalue_outlier] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_row, ...
                                                          rows, cols, vals, plinear_row, unlab, onen_un);
    %% Add al to form the final result
    Rind      = [  Rind_lbn;  Rind_outlier];
    Rvalue    = [Rvalue_lbn;Rvalue_outlier];
    y_new     = [y_new_pinitl;y_new_punlabeled];
    x_st_new  = [x_st_new_pinitl ; x_st_new_punlab];
    size_of_s = size_of_s_labeled_notstart + size_of_s_unlab;
    constraint_to_instance_map = [constraint_to_instance_map_initL;constraint_to_instance_map_unlab];
    s         = c_mul_pAndw_o*[s_labeled_notstart;s_unlab]/100;
end