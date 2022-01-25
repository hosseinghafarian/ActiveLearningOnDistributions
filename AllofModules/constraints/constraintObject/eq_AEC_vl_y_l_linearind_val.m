function [Rind, Rvalue, b_EC, constraint_to_instance_map, y_new, size_of_b, nzmax]     = eq_AEC_vl_y_l_linearind_val(pre_constr_inst_map, pre_y, c_mul_pAndw_o,e)
    % constraint: v_l = y_l
    % b_EV = Yl
    %A_EV = sparse(nSDP*nSDP+3*n,n_l);
global cnstData
global cnstDefs
    nSDP  = cnstData.nSDP;
    initL = cnstData.initL(cnstData.initLnozero)';
    initLStart          = cnstData.initLStart'; 
    initLStart_notnoisy = cnstData.initLStart_notnoisy;
    if initLStart_notnoisy 
       initialinst      = setdiff(initL, initLStart);
       init_start       = initLStart';
    else
       initialinst      = initL;
       init_start       = [];
    end
    n_li   = numel(init_start);
    onenli = ones(1,n_li);
    rows   = [     init_start,  nSDP*onenli];
    cols   = [    nSDP*onenli, init_start];
    vals   = 0.5*[onenli,onenli];
    linear_row  = [1:n_li,1:n_li];
    nzmax  = 2;
    y_new  = zeros(n_li,1);
    if ~isempty(pre_constr_inst_map)
        ind  = pre_constr_inst_map(:,3)==cnstDefs.CSTR_ID_ABSV_VL_YL_WO & ismember(pre_constr_inst_map(:,2),init_start);
        pre_size = sum(ind);
        y_new(1:pre_size) = pre_y(ind);
    end
    constraint_to_instance_map = [repmat(cnstDefs.SINGLE_INS,n_li,1),init_start', repmat(cnstDefs.CSTR_ID_ABSV_VL_YL_WO,n_li,1)];
    [Rind , Rvalue] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_row, ...
                                rows, cols, vals, [], [], 0);
    b_EC  = c_mul_pAndw_o*cnstData.Yl(init_start);
    size_of_b = n_li;
end