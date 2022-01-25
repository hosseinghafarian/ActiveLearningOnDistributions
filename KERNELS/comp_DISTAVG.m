function [DISTX, DISTK, F_to_ind_map] = comp_DISTAVG(data, learningparams)
[DISTX, F_to_ind_map] = comp_DISTAVGX(data);
gamma    = learningparams.KOptions.gamma;
PD       = - pdist2(DISTX',DISTX');
DISTK    = exp(0.5*gamma*PD) ;
end
function [X, F_to_ind_map] = comp_DISTAVGX(data)
fuq = unique(data.F);
n   = numel(fuq);
F_to_ind_map = zeros(n,1);
X   = zeros(data.d, n);
for t = 1:n
    Fi = data.F==fuq(t);
    X(:,t) = mean(data.X(:,Fi),2);
    F_to_ind_map(t) = fuq(t);
end
end