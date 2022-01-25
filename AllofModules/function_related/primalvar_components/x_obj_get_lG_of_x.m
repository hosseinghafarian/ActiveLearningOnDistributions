function [l_x, G_x]              = x_obj_get_lG_of_x(x_k)
        G_x       = G_of_x(x_k);
        l_x       = l_of_x(x_k);
end