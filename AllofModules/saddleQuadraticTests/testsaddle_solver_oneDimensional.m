function testsaddle_solver 
max_iter = 100;
conv_meas= 10^-4;

x_0 = 1; 
u_0 = 1;
L_xx = 1/2;
L_uu = 3/2;
    function [grad_x, Hessian_xx, grad_u, Hessian_uu, Hessian_xu ] =second_order(y,v)
       grad_x     = 1/2*y-1/4*v^2;
       Hessian_xx = 1/2;
       grad_u     = -1/2*(y+1)*v-1/2*(v-1);
       Hessian_uu = -3/2*v;
       Hessian_xu = -1/2*v;
    end
    function [y,v] = project(x,u)
        y         = max(x, -1);
        y         = min(y, +1);
        v         = max(u, -1);
        v         = min(v, +1);
    end
[y,v] = saddle_solver(x_0, u_0, @second_order, @project, L_xx, L_uu, max_iter, conv_meas);

x_0 = 0.75; 
u_0 = 0.68;
L_xx = 2;
L_uu = 1/2;
    function [grad_x, Hessian_xx, grad_u, Hessian_uu, Hessian_xu ] =second_order2(y,v)
       grad_x     = 1-1/2*v+2*y;
       Hessian_xx = 2;
       grad_u     = -1/2*y - v;
       Hessian_uu = -1;
       Hessian_xu = -1/2;
    end
    function [y,v] = project2(x,u)
        y         = max(x, -1);
        y         = min(y, +1);
        v         = max(u, -1);
        v         = min(v, +1);
    end
[y,v] = saddle_solver(x_0, u_0, @second_order2, @project2, L_xx, L_uu, max_iter, conv_meas);

end