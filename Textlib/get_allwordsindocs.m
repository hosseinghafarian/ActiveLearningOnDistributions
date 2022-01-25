function [words] = get_allwordsindocs(bagofwords)
n = size(bagofwords, 1);
words = {};
k     = 1;
for i = 1:n
   [bagofw, n_i] = conv_bag_of_word(bagofwords{i}); 
   for j=1:n_i
      if numel(bagofw{j}) > 0 
         words{k} = bagofw{j};
         k        = k + 1;
      end
   end
end
words = unique(words);
end