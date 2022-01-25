function [lp_list] = select_subset(learning_param_list, profile)
    n_lp      = numel(learning_param_list);
    not_lt    = min(profile.CV_search_notlessthan, n_lp);
    if profile.Full_CV_search 
        inds  = 1:n_lp;
    else
        k     = floor(profile.CV_search_percent*n_lp/100);
        k     = max(k, not_lt);
%         k     = min(k, profile.CV_search_notmorethan);
        inds  = randperm(n_lp,k);
    end
    lp_list   = learning_param_list(inds);
end