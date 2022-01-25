function [ind] = queryrandom_instances(K, X, unlabeled, labeled, y_l,  UN_subsampling_settings)
% This function returns a random UN_subsampling_settings.percent of unlabeled instances.  
   nu    = numel(unlabeled);
   usize = floor(nu*UN_subsampling_settings.percent/100);
   indun   = sort(randperm(nu,usize)); 
   ind   = unlabeled(indun);
end