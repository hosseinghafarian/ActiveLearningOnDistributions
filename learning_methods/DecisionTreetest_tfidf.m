function [decision_values] = DecisionTreetest_tfidf(model, learningparams, data, idx)
xtest  = data.X_tfidf(:,idx);
global cnstDefs
decision_values  = predict(model.nb, xtest');
end