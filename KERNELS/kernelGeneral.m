function [K, dm, F_to_ind_row ] = kernelGeneral(learningparams, data, idx)
    % Transpose data to make every data columnize
    % And Append one to every x instead of b in w^Tx+b
    % Select Test indices
    if ~data.isDistData
        if nargin == 2
          idx = 1:data.n; 
        end
        [K, dm, F_to_ind_row, F_to_ind_col] = kernelArrayGeneral(data, idx, data, idx, learningparams, true);
    else
       if nargin == 2
          idx = unique(data.F); 
       end 
       [K, dm, F_to_ind_row, F_to_ind_col] = kernelArrayGeneral(data, idx, data, idx, learningparams, true);        
    end
end