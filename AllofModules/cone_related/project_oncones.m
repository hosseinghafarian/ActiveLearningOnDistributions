function [x_k]  = project_oncones(x_k, s_I)
    x_k  = project_u_of_x(x_k);
    x_k  = project_st_of_x(x_k, s_I);
end