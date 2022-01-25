function [ dm, F_to_ind_map_i, F_to_ind_map_j] = distance_matrix(data_i, Fidx_i, data_j, Fidx_j, gamma_is, iseq)
if nargin == 5
    iseq = false;
end
uF_i = unique(data_i.F);
if ~isempty(Fidx_i) %if isempty compute all 
   uF_i = intersect(uF_i, Fidx_i);
end
is_in_X_i = ismember(data_i.F, uF_i);
X_i  = data_i.X(:,is_in_X_i);
n   = numel(uF_i);
F_i = data_i.F(is_in_X_i);


%[dm3,  F_to_ind_map_i,  F_to_ind_map_j] = compute_usingMatlab();
if iseq
    tic
   [dm , F_to_ind_map_i]         = distmatrix_of_distributions_sameset(X_i, F_i, gamma_is);
   toc
   F_to_ind_map_j = F_to_ind_map_i;
else
   uF_j = unique(data_j.F);
   if ~isempty(Fidx_j)%if isempty compute all 
      uF_j = intersect(uF_j, Fidx_j); 
   end
   is_in_X_j = ismember(data_j.F, uF_j);
   X_j  = data_j.X(:,is_in_X_j); 
   F_j  = data_j.F(is_in_X_j);
   
   [dm, F_to_ind_map_i, F_to_ind_map_j] = distmatrix_of_distributions(X_i, F_i, X_j, F_j, gamma_is);
end
function [dm,  F_to_ind_map_i,  F_to_ind_map_j] = compute_usingMatlab()
dF_i = diff(F_i);
dF_j = diff(F_j);
idx_dF_i = [find(dF_i),numel(F_i)];
idx_dF_j = [find(dF_j),numel(F_j)];
stidx_F_i = 0;

uF_i = unique(data_i.F);
uF_i = intersect(uF_i, Fidx_i);
n   = numel(uF_i);
F_to_ind_map_i = zeros(n,1);
%if ~iseq 
    uF_j       = unique(data_j.F);
    uF_j = intersect(uF_j, Fidx_j);
    m              = numel(uF_j);
    F_to_ind_map_j = zeros(m,1);
% else
%     m = n;
% end
dmtic1=tic;
dm  = zeros(n,m);
if ~iseq
    for t = 1:n
       F_to_ind_map_i(t) = uF_i(t);
       for u = 1:m
          F_to_ind_map_j(u) = uF_j(u);
          dm(t, u) = NormDiffDistribution(data_i.X(:,data_i.F==uF_i(t)), data_j.X(:,data_j.F==uF_j(u)), gamma_is); 
       end
    end
    toc(dmtic1)
else    
    for t = 1:n
       endidx_F_i = idx_dF_i(t);
       X_it = data_i.X(:,stidx_F_i+1:endidx_F_i);
       F_to_ind_map_i(t) = uF_i(t);
       stidx_F_j = idx_dF_j(t);
       for u = t+1:n
          endidx_F_j = idx_dF_j(u);
          X_ju = data_j.X(:,stidx_F_j+1:endidx_F_j); 
          dm(t, u) = NormDiffDistribution(X_it, X_ju, gamma_is);%norm_diff_of_distribution
          stidx_F_j = endidx_F_j;
       end
       stidx_F_i = endidx_F_i;
    end
    dm = (dm+dm');
    F_to_ind_map_j = F_to_ind_map_i;
end
end
end

