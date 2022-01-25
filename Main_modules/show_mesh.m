function show_mesh(learning_params_list, learning_param_fields, AVG_param, z_label, save_figfilename)
global cnstDefs
[X, Y, Z] = get_field_data_in_Matrix(learning_params_list, learning_param_fields, AVG_param);
figure;
mesh(X, Y, Z);
ylabel(learning_param_fields.firstfield);xlabel(learning_param_fields.secondfield); zlabel(z_label);
savefig([cnstDefs.result_classification_path, save_figfilename,'_',datestr(datetime(),'yy_mm_dd_HH_MM')]);
end