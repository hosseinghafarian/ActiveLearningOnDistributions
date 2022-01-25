function [decision_values] = KNNtest_tfidf(model, learningparams, data, idx)
xtest  = data.X_tfidf(:,idx);
global cnstDefs
decision_values  = predict(model.nb, xtest');
end