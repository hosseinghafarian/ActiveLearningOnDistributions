function [acc_noise_ca_cp] = get_field_data(learning_params_list, learning_param_fields, AVG_param_results)
n = numel(learning_params_list);
acc_noise_ca_cp = zeros(n,4);
    
for i= 1:n
    lp = learning_params_list{i};
    if ~isempty(learning_param_fields.firstfield_part1)
       acc_noise_ca_cp(i,1) = lp.(learning_param_fields.firstfield_part1).(learning_param_fields.firstfield);
    else
       acc_noise_ca_cp(i,1) = lp.(learning_param_fields.firstfield);    
    end
    if ~isempty(learning_param_fields.secondfield_part1)
       acc_noise_ca_cp(i,2) = lp.(learning_param_fields.secondfield_part1).(learning_param_fields.secondfield);
    else
       acc_noise_ca_cp(i,2) = lp.(learning_param_fields.secondfield);    
    end
    acc_noise_ca_cp(i,3) = AVG_param_results(i);
end
end