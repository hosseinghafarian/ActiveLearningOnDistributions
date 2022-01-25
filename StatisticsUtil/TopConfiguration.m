function [indextop] = TopConfiguration(P_p, alphafast, K)
cochran = false;
P_m        = sum(P_p, 2)/size(P_p, 2);
[~, ind_sort]   = sort(P_m, 'ascend');
alphatilde = alphafast/(K-1);
for k = 2:K
    if cochran
       [h,p,stats] = cochranqtest(P_p(ind_sort(1:k),:)', alphafast);
    else
       p = friedman(P_p(ind_sort(1:k),:)', 1, 'off');
    end
    if p <alphatilde 
         break;
    end
end
indextop = ind_sort(1:k-1);
end