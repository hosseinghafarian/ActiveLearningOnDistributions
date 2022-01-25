function [first_subset, second_subset] = divide_crisis_data_by_Fid(data, Trainnumid, Testnumid, TR_subsampling_method, TR_subsampling_settings)
comp_mean = true;
second_subset = select_subset_place(data, Testnumid);
second_subset.X_tfidf  = data.X_tfidf(:, Testnumid);
second_subset.textof_tweets = data.textof_tweets(Testnumid);
second_subset.label_text    = data.label_text(Testnumid);
second_subset.X_pcaappend  = data.X_pcaappend(:, Testnumid);
checkdata(second_subset);
if comp_mean
    second_subset =  comp_mean_vec_dist(second_subset);
    second_subset.has_mean_vec = true;
end
second_subset.has_mean_vec = comp_mean;


odata                  = select_subset_place(data, Trainnumid);
checkdata(odata);
if nargin==5
   [data_tr]  = TR_subsampling_method( odata, TR_subsampling_settings); 
else
   data_tr    = odata;
end
data_tr.n              = numel(data_tr.Y);
first_subset           = data_tr;
first_subset.X_tfidf   = data_tr.X_tfidf(:, Trainnumid);
first_subset.textof_tweets = data.textof_tweets(Trainnumid);
first_subset.label_text    = data.label_text(Trainnumid);
first_subset.X_pcaappend   = data_tr.X_pcaappend(:, Trainnumid);
checkdata(first_subset);
if comp_mean
    first_subset =  comp_mean_vec_dist(first_subset);
    first_subset.has_mean_vec = true;
end
first_subset.has_mean_vec = comp_mean;
end