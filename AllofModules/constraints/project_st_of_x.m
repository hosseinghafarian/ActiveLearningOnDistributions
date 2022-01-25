function [x_k]  = project_st_of_x(x_k, s_I)
   x_k.st    = min(x_k.st,s_I);
end