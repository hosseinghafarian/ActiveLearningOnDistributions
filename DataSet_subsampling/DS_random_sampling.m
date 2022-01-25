function [data_s ] = DS_random_sampling(data,  DS_settings_num)
% Do nothing, just pass input to output. 
per   = DS_settings_num.percent;
n     = numel(data.Y);
ssize = floor((n*per)/100);
indperm   = randperm(n,ssize);
ind       = ismember(indperm, 1:n);
data_s            = data;
data_s.X          = data.X(:,ind);
data_s.F          = data.F(ind);
data_s.F_id       = data.F_id(ind);
data_s.Y          = data.Y(ind);
data_s.noisy      = data.noisy(ind);
data_s.labelnoise = data.labelnoise(ind);
data_s.n          = ssize;
end