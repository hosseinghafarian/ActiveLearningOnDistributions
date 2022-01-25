function [method_funcs, method_ind, found] = get_method_funcs(almethod_list, method_ID)
    n_e                   = numel(almethod_list);
    found                 = false;
    for i= 1:n_e
        if almethod_list{i}{7} == method_ID
            method_funcs = almethod_list{i};
            found             = true;
            method_ind        = i;
            return 
        end
    end
end