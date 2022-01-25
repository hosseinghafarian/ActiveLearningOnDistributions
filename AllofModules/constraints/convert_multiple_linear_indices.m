function [linearind, value] = convert_multiple_linear_indices(nzMax, nRows, nCols, linear_rows, mat_rows, mat_cols, matvals, plinear_rows, pag_inds, pagvals)
    % This function assumes a format linear(Matrix)+ pag, format and
    % convert indices based on it. if mat_row and mat_col was zero, then
    % it sets value for pag. Multiple elements in all of (linear(Matrix)+
    % pag) can have a value other than zero. number of elements in matvals and pagvals, can be inequal.  
    % nzMax is the maximum number of nonzero entries in a row. 
    uniquelin_row= unique(linear_rows);
    n_ur         = numel(uniquelin_row);
    linearind    = zeros(n_ur,nzMax);
    value        = zeros(n_ur,nzMax);
    matlinearind = sub2ind([nRows, nCols], mat_rows, mat_cols);
    offsetofpag  = nRows*nCols;
    nnzpvalzero  = nnz(pagvals);
    nnzmatvals   = nnz(matvals);
    if nnzpvalzero~=0 && nnzmatvals~=0
        for k=uniquelin_row
           plinind              = pag_inds(plinear_rows==k)+offsetofpag;
           pvals                = pagvals(plinear_rows==k);
           matind               = matlinearind(linear_rows==k);
           nnzpval              = nnz(pvals);
           n_colsused           = numel(matind)+nnzpval;
           if nnzpval~=0
              linearind(k,1:n_colsused) = [matind,plinind];
              value(k,1:n_colsused)           = [matvals(linear_rows==k),pvals];
           else
              linearind(k,1:n_colsused) = matind;
              value(k,1:n_colsused)           = matvals(linear_rows==k);
           end
        end
    elseif nnzmatvals~=0 && nnzpvalzero==0
        for k=uniquelin_row
           matind               = matlinearind(linear_rows==k);
           n_colsused           = numel(matind);
           linearind(k,1:n_colsused) = matind;
           value(k,1:n_colsused)           = matvals(linear_rows==k);
        end
    elseif nnzmatvals==0 && nnzpvalzero~=0
        for k=unique(plinear_rows)
           plinind              = pag_inds(plinear_rows==k)+offsetofpag;
           pvals                = pagvals(plinear_rows==k);
           n_colsused           = numel(plinind);
           linearind(k,1:n_colsused) = plinind;
           value(k,1:n_colsused)     = pvals;
        end
    else%if nnzmatvals==0 && nnzpvalzero==0
        linearind = [];
        value     = [];
    end
end