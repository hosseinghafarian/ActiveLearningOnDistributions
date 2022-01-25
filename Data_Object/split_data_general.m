function [first_subset, second_subset] = split_data_general(data, Trainnumid, Testnumid, TR_subsampling_method, TR_subsampling_settings)
second_subset = select_subset_place(data, Testnumid);
checkdata(second_subset);


odata       = select_subset_place(data, Trainnumid);
checkdata(odata);
if nargin==5
   [data_tr] = TR_subsampling_method( odata, TR_subsampling_settings); 
else
   data_tr   = odata;
end
data_tr.n              = numel(data_tr.Y);
first_subset           = data_tr;
checkdata(first_subset);
end