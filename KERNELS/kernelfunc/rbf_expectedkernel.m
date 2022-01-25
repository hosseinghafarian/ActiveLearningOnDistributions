function [KA, dm, F_ind_row, F_ind_col] = rbf_expectedkernel(xtrain, Fidxtr, xtest, Fidxte, learningparams, gamma, iseq)
    [KA, F_ind_row, F_ind_col] = rbf_emp(xtrain, Fidxtr, xtest, Fidxte, learningparams.KOptions.gamma_is, iseq);
    dm    = KA;

function [ K, F_to_ind_map_i, F_to_ind_map_j] = rbf_emp(data_i, Fidx_i, data_j, Fidx_j, gamma_is, iseq)
if nargin == 5
    iseq = false;
end
% uF_i = unique(data_i.F);
% if ~isempty(Fidx_i) %if isempty compute all 
%    uF_i = intersect(uF_i, Fidx_i);
% end
% is_in_X_i = ismember(data_i.F, uF_i);
% X_i  = data_i.X(:,is_in_X_i);
% n   = numel(uF_i);
% F_i = data_i.F(is_in_X_i);
[X_i, F_i, uF_i] = get_dataindex(data_i, Fidx_i);

% % % % %[dm3,  F_to_ind_map_i,  F_to_ind_map_j] = compute_usingMatlab();
% % % % % if iseq
% % % % %     tic
% % % % %    [K , F_to_ind_map_i]         = distmatrix_of_distributions_sameset(X_i, F_i, uF_i, gamma_is);
% % % % %    toc
% % % % %    F_to_ind_map_j = F_to_ind_map_i;
% % % % % % else
%    uF_j = unique(data_j.F);
%    if ~isempty(Fidx_j)%if isempty compute all 
%       uF_j = intersect(uF_j, Fidx_j); 
%    end
%    is_in_X_j = ismember(data_j.F, uF_j);
%    X_j  = data_j.X(:,is_in_X_j); 
%    F_j  = data_j.F(is_in_X_j);
[X_j, F_j, uF_j] = get_dataindex(data_j, Fidx_j);    
[K, F_to_ind_map_i, F_to_ind_map_j] = rbf_emp_mex(X_i, F_i, X_j, F_j, gamma_is);
end
end