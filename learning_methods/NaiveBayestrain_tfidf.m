function [model] = NaiveBayestrain_tfidf(learningparams, data, idx) 
    assert(isfield(data, 'X_pcaappend'), 'Error no tfidf feature is available');
    x_labeled  = double(data.X_tfidf(:,idx))';
    y_labeled  = double(data.Y(idx));
    data_noise = data.noisy(idx); 
global cnstDefs
x_ig_feat = sum(abs(x_labeled), 1)~=0;
x_labeledfeat = x_labeled(:, x_ig_feat);
model.nb = NaiveBayes.fit(x_labeledfeat', y_labeled', 'Prior', 'uniform');
model.modelname = 'naivebayes';
model.x_ig_feat = x_ig_feat;
end