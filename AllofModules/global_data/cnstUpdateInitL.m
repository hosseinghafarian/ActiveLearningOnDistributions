function [initLcurrnexInd] = cnstUpdateInitL(queryInstance)
% This function doesn't change query and unlabeled instances 
global cnstData

initLcurrnexInd  = cnstData.n_l+1;
cnstData.initL(initLcurrnexInd:initLcurrnexInd+cnstData.batchSize-1) = queryInstance; 
initLcurrnexInd  = initLcurrnexInd + cnstData.batchSize;
cnstData.n_l     = cnstData.n_l    + cnstData.batchSize;

end