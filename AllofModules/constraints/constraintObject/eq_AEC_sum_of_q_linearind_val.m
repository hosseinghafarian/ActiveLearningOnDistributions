function [Rind, Rvalue, b, constraint_to_instance_map, y_new, size_of_b, nz_max]  = eq_AEC_sum_of_q_linearind_val(pre_const_to_inst_map, pre_y, c_pwo, elements_of_diag)
% cConstraint= [cConstraint,sum(q)==batchSize];% constraints on q 
% Constraint :1^T q = bSize
%n_q       = numel(cnstData.extendInd);
global cnstData
global cnstDefs
    nSDP      = cnstData.nSDP;
    n_q       = cnstData.n_q;
    one2q     = ones(1,2*n_q);
    q_ind     = [repmat(nSDP,n_q,1),cnstData.extendInd'];  % this is the indexes of q
    rows      = [q_ind(:,1)',q_ind(:,2)'];
    cols      = [q_ind(:,2)',q_ind(:,1)'];
    linear_row= one2q;
    nz_max    = 2*n_q;
    vals      = 0.5*one2q;
    if ~isempty(pre_const_to_inst_map)
        y_new     = pre_y(pre_const_to_inst_map(:,3)== cnstDefs.CSTR_ID_SUMQ);
    else
        y_new     = 0;
    end
    constraint_to_instance_map =[ cnstDefs.EXTIND_INS, 0, cnstDefs.CSTR_ID_SUMQ];  
    [Rind , Rvalue] = convert_multiple_linear_indices(nz_max, nSDP, nSDP, linear_row, ...
                                rows, cols, vals, linear_row, linear_row, zeros(2*n_q,1));
    b         = c_pwo * cnstData.batchSize;
    size_of_b = 1;
    %row_size  = 2*n_u;
end