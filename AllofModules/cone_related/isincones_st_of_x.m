function [is_in]  = isincones_st_of_x(x_k, s_I)
   tol   = 10^-4;
   is_in = false;
   diff  = x_k.st-s_I;
   if (diff<tol)
       is_in = true;
   end
end