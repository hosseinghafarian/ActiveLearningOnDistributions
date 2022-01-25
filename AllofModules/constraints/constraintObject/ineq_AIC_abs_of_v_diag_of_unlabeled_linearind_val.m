function [Rind, Rvalue, s, constraint_to_instance_map, y_new, x_st_new, size_of_s,nzmax]                = ineq_AIC_abs_of_v_diag_of_unlabeled_linearind_val(pre_constr_inst_map_IC, pre_y,pre_x_st, c,p)
    % In the following r==G_plus(setunlab,setunlab)
    % cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
global cnstData
global cnstDefs
    nSDP     = cnstData.nSDP;
    unlab    = cnstData.unlabeled;
    n_u      = numel(unlab);
    oneu     = ones(1,n_u);
    halvec   = 0.5*oneu;
    nzmax    = 5;
    % Constraint: a= 1-p-q>= v :=> v-a<=0 :=> G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
    n_pe     = n_u;
    rows     = [unlab               , repmat(nSDP, 1, n_u),            unlab];  
    cols     = [repmat(nSDP, 1, n_u),                unlab,            unlab];
    vals     = [             halvec,                halvec,            -oneu];
    linear_rows = [1:n_u, 1:n_u, 1:n_u];    
    y_new_pe    = zeros(n_pe,1); 
    x_st_new_pe = zeros(n_pe,1);
    if ~isempty(pre_constr_inst_map_IC)
        % select those instances which is also in unlab. That is after
        % change of unlabeled instances (due to querying a batch of labels)
        % it is still part of unlabeled data and this set. So, we can reuse
        % its lagrange variable
        ind  = pre_constr_inst_map_IC(:,3)==cnstDefs.CSTR_ID_ABSV_DIAG_UNLABELED_PE&ismember(pre_constr_inst_map_IC(:,2),unlab);
        pre_size = sum(ind);
        y_new_pe(1:pre_size)    = pre_y(ind);
        x_st_new_pe(1:pre_size) = pre_x_st(ind);
    end
    constraint_to_instance_map_pos = [ repmat(cnstDefs.SINGLE_INS, n_u,1),unlab', repmat(cnstDefs.CSTR_ID_ABSV_DIAG_UNLABELED_PE,n_u,1)];
    [Rind_pe , Rvalue_pe] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:n_u, unlab, zeros(1,n_u));
    
    % Constraint: a= 1-p-q>= -v :=> -v-a<=0 :=> -G_plus(setunlab, nap+1)-G_plus(setunlab,setunlab)<=0 
    vals     = [            -halvec,               -halvec,            -oneu];
    n_ne      = n_u;
    y_new_ne    = zeros(n_ne,1); 
    x_st_new_ne = zeros(n_ne,1);
    if ~isempty(pre_constr_inst_map_IC)
        % select those instances which is also in unlab. That is after
        % change of unlabeled instances (due to querying a batch of labels)
        % it is still part of unlabeled data and this set. So, we can reuse
        % its lagrange variable
        ind  = pre_constr_inst_map_IC(:,3)==cnstDefs.CSTR_ID_ABSV_DIAG_UNLABELED_NE&ismember(pre_constr_inst_map_IC(:,2),unlab);
        pre_size = sum(ind);
        y_new_ne(1:pre_size)    = pre_y(ind);
        x_st_new_ne(1:pre_size) = pre_x_st(ind);
    end
    constraint_to_instance_map_neg = [ repmat(cnstDefs.SINGLE_INS, n_u,1),unlab', repmat(cnstDefs.CSTR_ID_ABSV_DIAG_UNLABELED_NE,n_u,1)];
    [Rind_ne , Rvalue_ne] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:n_u, unlab, zeros(1,n_u));
    Rind      = [  Rind_pe(1:n_pe,:);   Rind_ne(1:n_ne,:)];
    Rvalue    = [Rvalue_pe(1:n_pe,:); Rvalue_ne(1:n_ne,:)];
    y_new     = [           y_new_pe;            y_new_ne];
    x_st_new  = [ x_st_new_pe ; x_st_new_ne ];
    size_of_s = n_pe + n_ne;
    constraint_to_instance_map = [constraint_to_instance_map_pos;constraint_to_instance_map_neg];
    s         = zeros(size_of_s,1);
end