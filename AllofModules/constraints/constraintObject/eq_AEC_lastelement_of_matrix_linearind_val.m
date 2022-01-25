function [Rind , Rvalue, b, constraint_to_instance_map, y_new, size_of_b, nz_max]            = eq_AEC_lastelement_of_matrix_linearind_val(pre_const_to_inst_map, pre_y,c_pwo,elements_of_diag)  
    % cConstraint= [cConstraint,G_plus(nap+1,nap+1)==1];
    % Constraint: G(nSDP,nSDP) = 1
global cnstData
global cnstDefs
    nSDP      = cnstData.nSDP;
    rows      = nSDP;
    cols      = nSDP;
    linear_row= 1;
    vals      = 1;
    if ~isempty(pre_const_to_inst_map)
        y_new     = pre_y(pre_const_to_inst_map(:,3)== cnstDefs.CSTR_ID_LASTEL);
    else
        y_new     = 0;
    end
    constraint_to_instance_map =[cnstDefs.NO_INS, 0, cnstDefs.CSTR_ID_LASTEL]; 
    [Rind , Rvalue] = convert_multiple_linear_indices(1, nSDP, nSDP, linear_row, ...
                                rows, cols, vals, linear_row, linear_row, 0);
    b         = c_pwo;
    size_of_b = 1;
    nz_max    = 1;
end