function [decision_values] = DecisionTreetest_word2vecseriesappend(model, learningparams, data, idx)
xtest  = data.X_pcaappend(:,idx);
global cnstDefs
decision_values  = predict(model.nb, xtest');
end