function [n_l, Lindex, Uindex, Lind, Uind] = get_indices( data, samples_to_query_from, initL)
%get_indices This function returns indices of labeled and unlabeled data
%based on the instance IDs in input arguments
%   F_to_ind, F_id, samples_to_query_from, initL: all contain dist IDs,
%   data.F_to_ind: contains original indices of data before splitting to
%   train and test. 
%   this function gets labeled and unlabeled indices
%   this function works in relation to get_distID_fromind
unlabeled  = setdiff ( samples_to_query_from, initL)';
Uindex     = ismember(data.F_to_ind, unlabeled);
initL      = initL(initL>0)';
Lindex     = ismember(data.F_to_ind, initL);
n_l        = numel(initL);

Lind       = find(Lindex);
Uind       = find(Uindex);
end

