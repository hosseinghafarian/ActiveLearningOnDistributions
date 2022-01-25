function [ X_tfidf, X_wordId_doc, allwordsindoc] = get_tfidf(bagofwords)

[allwordsindoc] = get_allwordsindocs(bagofwords);
[X_wordId_doc]  = get_word_doc_mat(bagofwords, allwordsindoc);
X_tfidf         = tfidf(X_wordId_doc);

end