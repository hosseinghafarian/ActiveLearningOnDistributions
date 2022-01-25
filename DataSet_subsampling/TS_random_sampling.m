function [data] = TS_random_sampling(data,  DS_settings_num)
noise_num     = sum(data.noisy);
noise_percent = noise_num*100/numel(data.Y);
n     = numel(data.Y);
if DS_settings_num.method_to_adjust_size== 1 % use percentage for sampling
    per   = DS_settings_num.percent;
    ssize = floor((n*per)/100);
    nsize = floor(ssize*noise_percent/100);
    ind   = randperm(n-noise_num,ssize-nsize);
else                                                     % make size near to TS_random_sampling_settings.sample_if_larger_than
    ssize = DS_settings_num.sample_if_larger_than;
    if n <= ssize 
        return % Do nothing, just pass input to output.
    end
    nsize = floor(ssize*noise_percent/100);
    ind   = randperm(n-noise_num,ssize-nsize);
end    
noise_ind = find(data.noisy);
noise_sel = randperm(noise_num, nsize);
noise_isel= noise_ind(noise_sel);

nonnoise_ind = find(~data.noisy);
n_ind     = nonnoise_ind(ind);
ind       = union(n_ind, noise_isel);
selected  = false(1, n);
selected(ind) = true;
data.X          = data.X(:, selected);
data.Y          = data.Y(selected);
data.F          = data.F(selected);
data.noisy      = data.noisy(selected);
data.labelnoise = data.labelnoise(selected);
end