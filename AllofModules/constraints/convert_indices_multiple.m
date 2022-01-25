function [linearind, value] = convert_indices_multiple(nRows, nCols, mat_rows, mat_cols, matvals, pag_inds, pagvals)
    % This function assumes a format linear(Matrix)+ pag, format and
    % convert indices based on it. if mat_row and mat_col was zero, then
    % it sets value for pag. Multiple elements in all of (linear(Matrix)+
    % pag) can have a value other than zero. number of elements in matvals and pagvals, can be inequal.  
    matlinearind = sub2ind([nRows, nCols], mat_rows, mat_cols);
    paglinearind = nRows*nCols *ones(numel(pag_inds),1) + pag_inds;
    linearind    = [matlinearind;paglinearind];
    value        = [matvals;pagvals];
end