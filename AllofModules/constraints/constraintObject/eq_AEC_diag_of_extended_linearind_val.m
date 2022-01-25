function [Rind, Rvalue, b, constraint_to_instance_map, y_new, size_of_b,nz_max]= eq_AEC_diag_of_extended_linearind_val(pre_const_to_inst_map, pre_y,c_mul_pAndw_o,elements_of_diag)
    % cConstraint=[cConstraint, diag(G_plus(setQuery,setQuery))==q];
    % Constraint: diag(G_qq)==q
global cnstData
global cnstDefs
    extind   = cnstData.extendInd;
    n_e      = numel(extind);
    nSDP     = cnstData.nSDP;
    onevec   = ones(1,n_e);
    halvec   = -0.5*onevec;
    rows     = [extind,                extind, repmat(nSDP,1,n_e)];  
    cols     = [extind, repmat(nSDP, 1, n_e),              extind];
    vals     = [onevec, halvec, halvec];
    linear_rows = [1:n_e,1:n_e,1:n_e];
    nz_max   = 3;
    %% this is much different to warmstart lagrange variables. must consider original instances for each copy. we use function get_map_extendind to obtain the 
    % mapping from old y_? to new y_new
    if ~isempty(pre_const_to_inst_map)
        [eind_new, eind_pre] = get_map_extndind(cnstData.initL);
        assert(numel(eind_new)==n_e);
        ind_pre_extndind = pre_const_to_inst_map(:,3)== cnstDefs.CSTR_ID_DIAGQ_Q & ismember(pre_const_to_inst_map(:,2),eind_pre); 
        y_new     = pre_y(ind_pre_extndind);
    else
        y_new     = zeros(n_e,1);
    end
    size_of_b= n_e;
    constraint_to_instance_map =[ repmat(cnstDefs.EXTIND_INS, size_of_b,1), extind', repmat(cnstDefs.CSTR_ID_DIAGQ_Q, size_of_b,1) ]; 
    [Rind , Rvalue] = convert_multiple_linear_indices(nz_max, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:n_e, cnstData.query, zeros(n_e,1));
    b         = zeros(n_e,1);
end