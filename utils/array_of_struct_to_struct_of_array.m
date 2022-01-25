function [struct_of_array] = array_of_struct_to_struct_of_array(cellarray_of_struct, fields_list)
n           = numel(cellarray_of_struct);
if n==0, struct_of_array= struct();return;end
if ~isstruct(cellarray_of_struct{1}), struct_of_array= struct();return;end
if nargin == 1
   fields_list = fieldnames(cellarray_of_struct{1});
end
for i= 1:numel(fields_list)
    if isscalar(cellarray_of_struct{1}.(fields_list{i})) &&~isstruct(cellarray_of_struct{1}.(fields_list{i}))
        if isfield(cellarray_of_struct{1},fields_list{i})
           struct_of_array.(fields_list{i}) = get_array_of_fieldvalues(cellarray_of_struct,fields_list{i}); 
        end
    end
end
end