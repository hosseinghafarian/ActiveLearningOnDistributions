function cnstDataGetUnlabeled
global cnstData
    labeled   = initL;
    n_l       = size(labeled,1);
    aln_u     = cnstData.n_S - n_l;                % all unlabeled data
    % select only n_u points from unlabled data 
    n_u       = aln_u;                  % size of unlabeled data 
    unlabeleda= setdiff(1:n,labeled)';  % unlabeled data indices
    YfChecka  = yTrain(unlabeleda);    % unlabeled data lables 
    yuforsel  = unlabeleda.* YfChecka;  % indices of unlabled data .* labeles of them
    ulindexp  = yuforsel(yuforsel>=0); % indices of positive samples
    ulindexn  = yuforsel(yuforsel<0); % indices of positive samples

    ulindexn  = abs(ulindexn);          % use absolute value of negative instances
    unlabeled = sort([ulindexp;ulindexn]); % sort labeles of da
    YfCheck   = yTrain(unlabeled);
    queryDup  = unlabeled;
end