function [xTrain,yTrain,isnoisy,isoutlier,numofnoisy,lnoiseper,onoiseper] = makedatanoisy(xTrain, yTrain, percent, noisetype)
% this function will not be used any more
% This function makes data noisy ( contains label noise and/or outlier )
%    noisetype: 1 : label noise by flipping some labels.
n            = size(yTrain,1);
isnoisy      = false(n,1);
isoutlier    = false(n,1);
numofnoisy   = 0;
if noisetype == 0
    lnoiseper = 0;
    onoiseper = 0;
    return
elseif noisetype == 1 % this is label noise
    k            = round(n*percent/100);
    numofnoisy   = k;
    lnoiseper    = percent;
    onoiseper    = percent; % attention: onoiseper must always be greater than or equal to lnoiseper. 
    noiseindices = randperm(n,k);
    isnoisy(noiseindices) = true;
    yTrain(noiseindices)  = -yTrain(noiseindices);  % flip labels of noisy samples
end
end