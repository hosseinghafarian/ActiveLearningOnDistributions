function [unlabquery, unlabnotquery] = get_unlabeledAndquery()
%this function returns unlabeled samples which are not in query
global cnstData
unlabnotquery    = setdiff(cnstData.unlabeled,cnstData.query);
unlabquery       = intersect(cnstData.unlabeled,cnstData.query);
end