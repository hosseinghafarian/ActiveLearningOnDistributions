function [model] = NaiveBayestrain_word2vecseriesappend(learningparams, data, idx) 
    assert(isfield(data, 'X_pcaappend'), 'Error no tfidf feature is available');
    x_labeled  = double(data.X_pcaappend(:,idx))';
    y_labeled  = double(data.Y(idx));
    data_noise = data.noisy(idx); 
global cnstDefs
model.nb = NaiveBayes.fit(x_labeled, y_labeled', 'Prior', 'uniform');
model.modelname = 'naivebayes';
end