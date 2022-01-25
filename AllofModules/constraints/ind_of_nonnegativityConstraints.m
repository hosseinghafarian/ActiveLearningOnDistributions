function [ ind] = ind_of_nonnegativityConstraints(query, nSDP)
   ind_last= repmat(nSDP,numel(query),1);
   ind_s   = [query',ind_last;ind_last,query'];
   ind     = sub2ind([nSDP,nSDP],ind_s(:,1),ind_s(:,2));
end