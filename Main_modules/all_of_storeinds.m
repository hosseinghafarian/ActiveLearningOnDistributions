function [store_ind_union, method_store_inds, name_store_inds] = all_of_storeinds(experiment_sequence, almethod_list)
    n_e                   = numel(experiment_sequence);
    n_a                   = numel(almethod_list);
    method_store_inds     = zeros(4*n_e, 5);% max_store_len
    store_ind_union       = [];
    for j=1:n_a
        for i= 1:n_e
           if almethod_list{j}{7} == experiment_sequence{i}{1}
               store_len          = numel(almethod_list{j}{5});
               method_store_inds(j,1:store_len) = almethod_list{j}{5}';
               store_ind_union    = union(store_ind_union, almethod_list{j}{5});
               name_store_inds{j} = almethod_list{j}{6};
               break;
           end
        end
    end
end