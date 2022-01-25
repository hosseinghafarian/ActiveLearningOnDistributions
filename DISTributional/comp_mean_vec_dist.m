function [data] = comp_mean_vec_dist(data)

funq = data.F_id;
n    = numel(funq);
d    = size(data.X, 1);
data.X_avg = zeros(d, n);
for i = 1:n
   f_i = funq(i);
   data.X_avg(:,i) = comp_dist_avg(data, f_i);
end

function vec = comp_dist_avg(data, f_id)
ind    = data.F==f_id;
n_f_id = sum(ind);
if n_f_id 
   %XM     = data.X(:, ind);
   vec    = sum(data.X(:, ind), 2) / n_f_id;
else
   vec    = NAN;
end
end
end
