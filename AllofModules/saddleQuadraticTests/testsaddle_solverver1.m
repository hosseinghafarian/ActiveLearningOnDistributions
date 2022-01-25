function testsaddle_solverver1 
n_x       = 100000;
n_u       = 1000; 
strange   = -1;
enrange   = 1;
options.usexHessian = false;
options.useuHessian = false;
options.usexpcg     = false;
options.useupcg     = false;
options.pcgtolstart = 10^-1;
options.pcgtolend   = 10^-6;
options.useuL_xu    = true;
options.usexL_xu    = true;

max_iter  = 1000;
conv_meas = 10^-4;
range_st_x= 0.25;
range_en_x= 0.75;
x_star    = (range_en_x-range_st_x)*rand(n_x,1);
u_star    = (range_en_x-range_st_x)*rand(n_u,1);
x_reg     = x_star;
u_reg     = u_star;

x_0       = strange + (enrange-strange)*rand(n_x,1); 
u_0       = strange + (enrange-strange)*rand(n_u,1);
M         = strange/10 + 1/10*(enrange-strange)*rand(n_x,n_u);
mu_strcvx = 5;
%% When useuL_xu and usexL_xu is true then 
% strong convexity values larger than 9.75 when size is (d_x, d_u) = (1000,60) makes the algorithm non-convergent. Why? Where this value comes from?;
% strong convexity values smaller than 3.5 when size is (d_x, d_u) = (1000,60) makes the algorithm non-convergent. Why? Where this value comes from?;
%%
L_xx      = mu_strcvx;
L_uu      = mu_strcvx;
L_xu      = norm(M);
    function [fvalue ] = fxu(y,v)
       fvalue = mu_strcvx*norm(y-x_star)^2+ (y-x_star)'*M*(v-u_star)-mu_strcvx/2*norm(v-u_star)^2; 
    end
    function [Hessian_xx] = second_order_x(y,v)
       grad_x     = mu_strcvx*(y-x_reg)+M*(v-u_reg);
       Hessian_xx = mu_strcvx*eye(n_x);
       grad_u     = M'*(y-x_reg)-mu_strcvx*(v-u_reg);
       Hessian_uu = -mu_strcvx*eye(n_u);
       Hessian_xu = M;
    end
    function [Hessian_uu] = second_order_u(y,v)
       grad_x     = mu_strcvx*(y-x_reg)+M*(v-u_reg);
       Hessian_xx = mu_strcvx*eye(n_x);
       grad_u     = M'*(y-x_reg)-mu_strcvx*(v-u_reg);
       Hessian_uu = -mu_strcvx*eye(n_u);
       Hessian_xu = M;
    end
    function [grad_x, grad_u, Hessian_xu ] = semi_second_order(y,v)
       grad_x     = mu_strcvx*(y-x_reg)+M*(v-u_reg);
       grad_u     = M'*(y-x_reg)-mu_strcvx*(v-u_reg);
       Hessian_xu = M;
    end
    function [y,v] = project(x,u)
        y         = max(x, strange);
        y         = min(y, enrange);
        v         = max(u, strange);
        v         = min(v, enrange);
    end
    function [x_k, u_k] = prox_operator(L_xx, L_uu, L_xu, L_A_x, L_A_u, a_x, A_x, a_u, A_u, y, v, usexL_xu, useuL_xu)
        x_k      = y - (1/L_A_x)*a_x;
        u_k      = v - (1/L_A_u)*a_u; 
        [x_k,u_k]= project(x_k, u_k);
    end
    function [conv_meas] = compute_measure(meastype, x_k,u_k, y, v)
        conv_meas  = norm(y-x_k)+norm(v-u_k);
    end
function_arguments.semi_second_order = @semi_second_order;
function_arguments.Hessian_x         = @second_order_x;
function_arguments.Hessian_u         = @second_order_u;
function_arguments.prox_operator     = @prox_operator;
function_arguments.objective_func    = @fxu;
function_arguments.converge_measure  = @compute_measure;

[y,v] = saddle_solver_ver2(options, x_0, u_0, function_arguments, n_x, n_u, L_xx, L_uu, L_xu, max_iter, conv_meas);
diff  = norm(y-x_star)+norm(v-u_star);
end