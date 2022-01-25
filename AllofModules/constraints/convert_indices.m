function [linearind, value] = convert_indices(nRows, nCols, mat_row, mat_col, pag_ind, val)
    % This function assumes a format linear(Matrix)+ pag, format and
    % convert indices based on it. if mat_row and mat_col was zero, then
    % it sets value for pag. just one element in all of (linear(Matrix)+
    % pag) can have a value other than zero. 
    if mat_row~=0 && mat_col~=0
        linearind = sub2ind([nRows, nCols], mat_row, mat_col);
    else
        linearind = nRow*nCols + pag_ind;
    end
    value         = val;
end