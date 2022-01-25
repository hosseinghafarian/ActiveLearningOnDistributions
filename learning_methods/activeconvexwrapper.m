function [] = activeconvexwrapper(initL, unlabeled )
global cnstData
unlabeled         = cnstData.unlabeled;
if numel(unlabeled)==0 
            return 
end
Yl                = yTrain(initL');
[ALresult,model_set,Y_set]= ActiveDConvexRelax2(sw,ModelInfo,learningparams,initL,cnstData.unlabeled,Yl,cnstData.n_l,cnstData.n_u,batchSize,n_o,lnoiseper,onoiseper);   
tq                = k_mostlargest(qresult,batchSize);
qi                = unlabeled(tq); 
ALresult.queryInd = qi;

end