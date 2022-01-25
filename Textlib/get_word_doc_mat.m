function [X_wordId_doc] = get_word_doc_mat(bagofwords, allwordsindoc)
n_w = numel(allwordsindoc);
words_Id = containers.Map(allwordsindoc,(1:n_w));
X_wordId_doc = [];
n_bag = numel(bagofwords);
for i = 1:n_bag
   word_Id4bag  = zeros(n_w, 1);
   [bagofw, n_i] = conv_bag_of_word(bagofwords{i}); 
   for j=1:n_i
      if isKey(words_Id, bagofw{j})
          wid = words_Id(bagofw{j});
          word_Id4bag(wid) = word_Id4bag(wid) + 1;
      end    
   end
   X_wordId_doc = [ X_wordId_doc, word_Id4bag];
end
assert(numel(bagofwords)==size(X_wordId_doc, 2), 'Error in get_word_doc_mat');
nowordinbags = sum(X_wordId_doc, 1)~=0;
X_wordId_doc = X_wordId_doc(:, nowordinbags);
end