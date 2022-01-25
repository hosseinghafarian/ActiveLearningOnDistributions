function [X, Y, Z] = get_field_data_in_Matrix(learning_params_list, learning_param_fields, AVG_param_results)

[AVG_param_field] = get_field_data(learning_params_list, learning_param_fields, AVG_param_results);
x = sort(unique(AVG_param_field(:,1)));
y = sort(unique(AVG_param_field(:,2)));
n_x    =numel(x);
n_y    =numel(y); 
Z      = zeros(n_x, n_y);
for i=1:n_x
    for j=1:n_y
        Z(i,j) = AVG_param_field(AVG_param_field(:,1)==x(i) & AVG_param_field(:,2)==y(j),3);
    end
end
[X,Y]= meshgrid(y,x);
end