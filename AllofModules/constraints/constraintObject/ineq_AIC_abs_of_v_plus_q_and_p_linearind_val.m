function [Rind , Rvalue, s, size_of_s, nzmax]       = ineq_AIC_abs_of_v_plus_q_and_p_linearind_val(c_mul_pAndw_o,e)
    %cConstraint= [cConstraint,0<=q], where is q<=1? q<=1 is not necessary
    %because of the constraint p+q <=1, all are positive. so, this constraint
    %is deleted. 
    % cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
    % Constraint: v <= a= 1-p-q :=> v+p+c_mul_pAndw_o*q <=1 
global cnstData
    ap       = ones(cnstData.n_S,1);
    n_u      = numel(cnstData.unlabeled);
    n_q      = numel(cnstData.extendInd);
    nSDP     = cnstData.nSDP;
    %% if all of unlabeled samples are not in the query, we deal with two subsets, one that is unlabeled, and candidate for query and one that is unlabeled and not candid for query
    [unlabquery, unlabnotquery] = get_unlabeledAndquery();
    n_uq     = numel(unlabquery);
    n_unq    = numel(unlabnotquery);
    halvecq  = 0.5*ones(1,n_uq);
    halvec   = 0.5*ones(1,n_unq);
    nzmax    = 5;
    % for instances candid for query
    rows     = [unlabquery           , repmat(nSDP, 1, n_uq),            cnstData.extendInd, repmat(nSDP,1,n_uq)  ];  
    cols     = [repmat(nSDP, 1, n_uq),            unlabquery,   repmat(nSDP,1,n_uq), cnstData.extendInd    ];
    vals     = [               halvecq,                halvecq, halvecq*c_mul_pAndw_o, halvecq*c_mul_pAndw_o];
    linear_rows = [1:n_uq,1:n_uq,1:n_uq,1:n_uq];
    [Rind_pe , Rvalue_pe] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:n_uq, unlabquery, ones(n_uq,1));
    n_pe     = n_uq;
    % for instances not candid for query                        
    rows     = [unlabnotquery           , repmat(nSDP, 1, n_unq)];  
    cols     = [repmat(nSDP, 1, n_unq),            unlabnotquery];
    vals     = [               halvec,                    halvec];
    linear_rows = [1:n_unq,1:n_unq];
    [Rind_pn , Rvalue_pn] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:n_unq, unlabnotquery, ones(n_unq,1));
    n_pn     = n_unq;
    %Constraint: -v - a<=0 :=>-v+p+c_mul_pAndw_o*q<=1
    halvecq  = 0.5*ones(1,n_uq);
    halvec   = 0.5*ones(1,n_unq);
            % for instances candid for query
    rows     = [unlabquery           , repmat(nSDP, 1, n_uq),            cnstData.extendInd, repmat(nSDP,1,n_uq)  ];  
    cols     = [repmat(nSDP, 1, n_uq),            unlabquery,   repmat(nSDP,1,n_uq), cnstData.extendInd    ];
    vals     = [              -halvecq,               -halvecq, halvecq*c_mul_pAndw_o, halvecq*c_mul_pAndw_o];
    linear_rows = [1:n_uq,1:n_uq,1:n_uq,1:n_uq];
    [Rind_ne , Rvalue_ne] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:n_uq, unlabquery, ones(n_uq,1));
    % for instances not candid for query                        
    rows     = [unlabnotquery           , repmat(nSDP, 1, n_unq)];  
    cols     = [repmat(nSDP, 1, n_unq),            unlabnotquery];
    vals     = [               -halvec,                  -halvec];
    linear_rows = [1:n_unq,1:n_unq];
    [Rind_nn , Rvalue_nn] = convert_multiple_linear_indices(nzmax, nSDP, nSDP, linear_rows, ...
                                rows, cols, vals, 1:n_unq, unlabnotquery, ones(n_unq,1));
    Rind   = [Rind_pe(1:n_pe,:)  ;    Rind_pn(1:n_pn,:);  Rind_ne(1:n_pe,:);  Rind_nn(1:n_pn,:)];
    Rvalue = [Rvalue_pe(1:n_pe,:);  Rvalue_pn(1:n_pn,:);Rvalue_ne(1:n_pe,:);Rvalue_nn(1:n_pn,:)];
    size_of_s = n_pe + n_pn + n_pe + n_pn;
    s      = ones(size_of_s,1);
end