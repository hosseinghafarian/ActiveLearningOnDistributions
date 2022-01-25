function [eind_new, eind_pre] = get_map_extndind(ind_to_delete)
global cnstData
ind1          = ~ismember(cnstData.pre_query_to_extend_map(:,1), ind_to_delete);
eind_pre      = cnstData.pre_query_to_extend_map(ind1,2);
ind2          = cnstData.query_to_extend_map(:,1)==cnstData.pre_query_to_extend_map(ind1,1);
eind_new      = cnstData.query_to_extend_map(ind2,2);
end