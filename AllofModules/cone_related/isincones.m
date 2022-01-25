function [is_in]  = isincones(x_k, s_I)
    u_is   = isincones_u_of_x(x_k);
    st_is  = isincones_st_of_x(x_k, s_I);
    is_in  = u_is & st_is;
end