function [Rind, Rvalue, s, constraint_to_instance_map, y_new, x_st_new, size_of_s, B_IV] = ineq_AIV_abs_of_wo_p_linearind_val(pre_constr_inst_map, pre_y, pre_x_st, c_mul_pAndw_o,e)

    % for k = cnstData.extendInd
    %     R = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
    %     A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
    %     s_IV(j,1) = 1;
    %     j = j + 1;
    % end

    % %A_IV  = [A_IV,A_IV];
    % for k = cnstData.extendInd
    %     R         = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
    %     A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
    %     s_IV(j,1)   = 1;
    %     j         = j + 1;
    % end
    % Constraint:  \phi(X)^T w_o  <= p
global cnstData
global cnstDefs
    n_S     = cnstData.n_S;
    nSDP    = cnstData.nSDP;
    
    nzmax   = 1;
    rows    = [];
    cols    = [];
    vals    = 0;
    linear_row  = 1:n_S;
    plinear_row = 1:n_S;
    y_new_pe    = zeros(n_S,1);
    x_st_new_pe = zeros(n_S,1);
    if ~isempty(pre_constr_inst_map)
        ind  = pre_constr_inst_map(:,3)==cnstDefs.CSTR_ID_ABS_WO_P_PE & ismember(pre_constr_inst_map(:,2),1:n_S);
        pre_size = sum(ind);
        y_new_pe(1:pre_size) = pre_y(ind);
        x_st_new_pe(1:pre_size) = pre_x_st(ind);
    end
    constraint_to_instance_map_pe = [repmat(cnstDefs.P_INS,n_S,1),(1:n_S)',repmat(cnstDefs.CSTR_ID_ABS_WO_P_PE,n_S,1)];
    [Rind_pe , Rvalue_pe] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_row, ...
                                rows, cols, vals, plinear_row, 1:n_S, -ones(1,n_S));
    Rind_ne   = Rind_pe;
    Rvalue_ne = Rvalue_pe;
    y_new_ne = zeros(n_S,1);
    x_st_new_ne = zeros(n_S,1);
    if ~isempty(pre_constr_inst_map)
        ind  = pre_constr_inst_map(:,3)==cnstDefs.CSTR_ID_ABS_WO_P_NE & ismember(pre_constr_inst_map(:,2),1:n_S);
        pre_size = sum(ind);
        y_new_ne(1:pre_size) = pre_y(ind);
        x_st_new_ne(1:pre_size) = pre_x_st(ind);
    end    
    constraint_to_instance_map_ne = [repmat(cnstDefs.P_INS,n_S,1),(1:n_S)',repmat(cnstDefs.CSTR_ID_ABS_WO_P_NE,n_S,1)];
    Rind      = [Rind_pe;Rind_ne];
    Rvalue    = [Rvalue_pe;Rvalue_ne];
    constraint_to_instance_map = [constraint_to_instance_map_pe;constraint_to_instance_map_ne];
    y_new     = [y_new_pe;y_new_ne];
    x_st_new  = [x_st_new_pe;x_st_new_ne];
    s         = zeros(2*n_S,1);
    size_of_s = 2*n_S;
    
    % Iind     = speye(cnstData.n_S,cnstData.n_S);
    % I_rep  = [sparse(eye_q);-sparse(eye_q);speye(cnstData.n_S);-speye(cnstData.n_S)];
    % B_IV   = (I_rep*cnstData.K)';
    I_rep  = [speye(n_S);-speye(n_S)];
    B_IV   = (I_rep*cnstData.K_o)';
end