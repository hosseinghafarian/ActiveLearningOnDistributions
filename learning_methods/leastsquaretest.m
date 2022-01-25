function [predict_label, accuracy, decision_values] = leastsquaretest(xtest, ytest, model)
    labelX = ytrain(initL);
    % Computing J(f,lambda) for labeled training data
    empRisk = labelX'*Klambdainv*labelX;
    % Does classifer correctly predicts queryInstance Label?
    KAq = KernelArrayNewInstance(isDistData,xTrain,xTrain(queryInstance),Options.KOptions);
    f=labelX'*Klambdainv*KAq{1}([initL]);
    Sf=sign(f);
    if (Sf==yTrain(queryInstance))
        Predict = true;
    else
        Predict = false;
    end
    Sf=size(yTest,1); 
    niseq=0;
    for it=1:size(xTest)
        KAl=ModelInfo.KA{it}(initL);
        f=labelX'*Klambdainv*KAl;
        Sf(t)=sign(f);
        niseq=niseq+isequal(sign(f),yTest(it));
    end
    accuracy = niseq/size(xTest,1); 
end