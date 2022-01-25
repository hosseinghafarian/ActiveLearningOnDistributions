function [queryind, q, model_set, Y_set]= activediversemultitask(sw,initL,Yl,lambda)
Yl  = yTrain(initL')';
[q,model_set,Y_set]= ActiveCRSemiSVM03(sw,initL,Yl,learningparams.lambda);
model.model = model_set;
model.type  = 4;               % 4 : diverse subtask
model.T     = size(model_set); % T : if diverse subtask
%% Setting Active Learning Results 
thresholdQ = 0.9;
[~,ql]=max(q);
%ql = (q > thresholdQ);
queryind = unlabeled(ql);  
end