function [learning_param_list, learning_param_fields] = make_learning_list(learning_params_ref, Classification_exp_param)
% This function makes list of fields list for cross validation for function
% simple-complex
start_p1          = Classification_exp_param.firstfield_start;
end_p1            = Classification_exp_param.firstfield_end;
step_p1           = Classification_exp_param.firstfield_step;
param_1_scale     = start_p1:step_p1:end_p1;
start_p2          = Classification_exp_param.secondfield_start;
end_p2            = Classification_exp_param.secondfield_end;
step_p2           = Classification_exp_param.secondfield_step;
param_2_scale     = start_p2:step_p2:end_p2;

firstfield_part1  = Classification_exp_param.firstfield_part1;
firstfield        = Classification_exp_param.firstfield;
firstfield_mul    = Classification_exp_param.firstfield_mul;
secondfield_part1 = Classification_exp_param.secondfield_part1;
secondfield       = Classification_exp_param.secondfield;
secondfield_mul   = Classification_exp_param.secondfield_mul;
i       = 1;
n_p1    = numel(param_1_scale);
n_p2    = numel(param_2_scale);
n_pl    = n_p1*n_p2;
learning_param_list = cell(n_pl,1);
for p_1 = 1:n_p1
   for p_2 = 1:n_p2
       learning_params_temp = learning_params_ref;
       % multiply parameter 1
       comp_field(firstfield_part1,   firstfield,  firstfield_mul, param_1_scale(p_1));
       % multiply parameter 2 
       comp_field(secondfield_part1, secondfield, secondfield_mul, param_2_scale(p_2));
       n =learning_params_ref.n;
       [learning_param_list{i} ]     = learning_settings(n, 'init', learning_params_temp);
       i    = i+ 1;
   end
end
learning_param_fields.firstfield_part1  = firstfield_part1;
learning_param_fields.firstfield        = firstfield;
learning_param_fields.secondfield_part1 = secondfield_part1;
learning_param_fields.secondfield       = secondfield;
    function comp_field(field_part1, field, field_mul, param_scval)
       if field_mul  
           if ~isempty(field_part1)
              learning_params_temp.(field_part1).(field)  = param_scval*learning_params_temp.(field_part1).(field);
           else
              learning_params_temp.(field)                = param_scval*learning_params_temp.(field_part1).(field);
           end 
       else
           if ~isempty(field_part1)
              learning_params_temp.(field_part1).(field)  = param_scval;
           else
              learning_params_temp.(field)                = param_scval;
           end
       end
    end
end
