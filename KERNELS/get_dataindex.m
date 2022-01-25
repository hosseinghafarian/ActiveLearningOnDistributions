function [X_i, F_i, uF_i] = get_dataindex(data_i, Fidx_i)
uF_i = unique(data_i.F);
if ~isempty(Fidx_i) %if isempty compute all 
   uF_i = intersect(uF_i, Fidx_i);
end
is_in_X_i = ismember(data_i.F, uF_i);
X_i  = data_i.X(:,is_in_X_i);
n   = numel(uF_i);
F_i = data_i.F(is_in_X_i);
end