function [bagofw, n_i] = conv_bag_of_word(textofi)
       c       = textscan(textofi,'%s','Delimiter', {' ','@','(', '''', ')', ':', '&', '*', ...
                                  ';', '!', '?', '+', '/','//','$', '#',',','-','.'});
       n_i     = numel(c{1,1});
       bagofw  = c{1,1}; 
 end