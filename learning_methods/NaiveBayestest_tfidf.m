function [decision_values] = NaiveBayestest_tfidf(model, learningparams, data, idx)
xtest  = data.X_tfidf(:,idx);
global cnstDefs
xtestfeat = xtest(:,model.x_ig_feat);
decision_values  = predict(model.nb, xtestfeat');
end