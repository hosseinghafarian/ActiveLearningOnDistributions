function myarray = get_array_of_fieldvalues(cellarray_of_struct,fields_name)
n       = numel(cellarray_of_struct);
myarray = zeros(n,1);
for i= 1:n
   if ~isempty(cellarray_of_struct{i})
       if isscalar(cellarray_of_struct{i}.(fields_name))
           myarray(i, 1) = cellarray_of_struct{i}.(fields_name); 
       end
   end
end
end