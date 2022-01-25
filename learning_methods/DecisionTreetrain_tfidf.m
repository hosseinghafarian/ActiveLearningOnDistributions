function [model] = DecisionTreetrain_tfidf(learningparams, data, idx) 
    assert(isfield(data, 'X_tfidf'), 'Error no tfidf feature is available');
    x_labeled  = double(data.X_tfidf(:,idx))';
    y_labeled  = double(data.Y(idx));
    data_noise = data.noisy(idx); 
global cnstDefs
model.nb = fitctree(x_labeled, y_labeled');
model.modelname = 'decisiontree';
end