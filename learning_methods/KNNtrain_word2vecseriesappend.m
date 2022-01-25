function [model] = KNNtrain_word2vecseriesappend(learningparams, data, idx) 
    assert(isfield(data, 'X_pcaappend'), 'Error no tfidf feature is available');
    x_labeled  = double(data.X_pcaappend(:,idx))';
    y_labeled  = double(data.Y(idx));
    data_noise = data.noisy(idx); 
global cnstDefs
if isfield(learningparams, 'numneighbors')
    numneighbors = learningparams.numneighbors;
else
    numneighbors = 7;
end
model.nb = fitcknn(x_labeled, y_labeled', 'NumNeighbors', numneighbors);
model.modelname = 'decisiontree';
end