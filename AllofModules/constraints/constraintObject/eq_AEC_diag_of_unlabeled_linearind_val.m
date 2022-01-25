function [Rind, Rvalue, b,  constraint_to_instance_map, y_new, size_of_b, nz_max]                   = eq_AEC_diag_of_unlabeled_linearind_val(pre_const_to_inst_map, pre_y, c,elements_of_diag)
    % this is equivalent to diag(G_{uu})+ p_u + q==1
    % if we want to select from all of unlabeled instances for query, it is ok.
    % But if some of unlabeled instances are not for querying, then it's not.
    % in this case we have diag(G_{uu}) + p_u == 1
    % so, we must divide the following to two subsets, one that is
    % cnstData.query, subset of unlabeled data for querying and one that is
    % not for querying. 
global cnstData
global cnstDefs
    nSDP     = cnstData.nSDP;
    queryind = cnstData.query;
    extndind = cnstData.extendInd;
    %% query subset of unlabeled data
    neq   = numel(queryind);
    onevec= ones(1,neq);
    halvec= 0.5*onevec;
    assert(numel(queryind) == numel(extndind));
    size_of_b_query    = neq;
    rows  = [queryind,             extndind, repmat(nSDP,1,neq)];  
    cols  = [queryind, repmat(nSDP,1 , neq),           extndind];
    vals  = [onevec, halvec, halvec];
    linear_rows = [1:neq,1:neq,1:neq];
    nz_max  = 4;
    y_new_query     = zeros(size_of_b_query, 1);
    if ~isempty(pre_const_to_inst_map)
        %find indices of instances which is a member of newqueryind and
        %also, was present in previous constraints the same as this. Also, compute number of them.  
        ind_pre_query = pre_const_to_inst_map(:,3)== cnstDefs.CSTR_ID_DIAG_P_Q_UNLAB_QUERY &ismember(pre_const_to_inst_map(:,2), queryind);
        pre_size_of_b = sum(ind_pre_query);
        y_new_query(1:pre_size_of_b) = pre_y(ind_pre_query);
    end
    constraint_to_instance_mapquery = [repmat(cnstDefs.SINGLE_INS,size_of_b_query,1), queryind', repmat(cnstDefs.CSTR_ID_DIAG_P_Q_UNLAB_QUERY,size_of_b_query,1)];
    [Rindquery , Rvaluequery] = convert_multiple_linear_indices(nz_max, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:neq, queryind, onevec);
    bquery     = elements_of_diag*onevec;
    %% nonquery subset of unlabeled data
    non_queryind  = setdiff(cnstData.unlabeled, queryind);
    size_of_b_nonquery = numel(non_queryind);
    nonevec= ones(1,size_of_b_nonquery);
    rows  = non_queryind;  
    cols  = non_queryind;
    vals  = nonevec;
    linear_rows = 1:size_of_b_nonquery;
    y_new_nonquery     = zeros(size_of_b_nonquery, 1);
    if ~isempty(pre_const_to_inst_map)
        %find indices of instances which is a member of newqueryind and
        %also, was present in previous constraints the same as this. Also, compute number of them.  
        ind_pre_noquery = pre_const_to_inst_map(:,3)== cnstDefs.CSTR_ID_DIAG_P_Q_UNLAB_NOQUERY &ismember(pre_const_to_inst_map(:,2), non_queryind);
        pre_size_of_b_noquery = sum(ind_pre_noquery);
        y_new_nonquery(1:pre_size_of_b_noquery) = pre_y(ind_pre_noquery);
    end
    constraint_to_instance_mapnonquery = [repmat(cnstDefs.SINGLE_INS, size_of_b_nonquery,1), non_queryind', repmat(cnstDefs.CSTR_ID_DIAG_P_Q_UNLAB_NOQUERY,size_of_b_nonquery,1)];
    [Rindnonquery , Rvaluenonquery] = convert_multiple_linear_indices(nz_max, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:size_of_b_nonquery, non_queryind, nonevec);
    bnonquery     = elements_of_diag*nonevec;
    %% ADD both constraints together.
    Rind   = [  Rindquery;  Rindnonquery];
    Rvalue = [Rvaluequery;Rvaluenonquery]; 
    b      = [     bquery,     bnonquery];
    constraint_to_instance_map = [constraint_to_instance_mapquery; constraint_to_instance_mapnonquery];
    y_new  = [y_new_query;y_new_nonquery];
    size_of_b                  = size_of_b_query+size_of_b_nonquery;
end