function [odata] = select_subset_place(data, placeindex)
odata            = data;
odata.F_id       = data.F_id(placeindex);
testismember     = ismember(data.F, odata.F_id);%, data.F);
odata.X          = data.X(:, testismember); 
odata.F          = data.F(:, testismember);
odata.Y          = data.Y(placeindex); 
odata.noisy      = data.noisy(placeindex);
odata.labelnoise = data.labelnoise(placeindex);
if odata.isTwoLevel
    odata.Y_L2        = data.Y_L2(testismember);
    odata.L2labelflag = data.L2labelflag(testismember);
end
odata.n          = numel(odata.Y);
end