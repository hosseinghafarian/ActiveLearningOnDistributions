function test2saddle_solverver1 

n_u       = 100; 
n_x       = n_u^2;
strange   = -1;
enrange   = 1;
options.usexHessian = false;
options.useuHessian = true;
options.usexpcg     = false;
options.useupcg     = false;
options.pcgtolstart = 10^-1;
options.pcgtolend   = 10^-6;
options.useuL_xu    = false;
options.usexL_xu    = true;

max_iter  = 1000;
max_conv_meas = 10^-4;
max_rel_gap   = 10^-3;
range_st_x= 0.25;
range_en_x= 0.75;
x_star    = (range_en_x-range_st_x)*rand(n_x,1);
x_Mat     = Mat(x_star);
x_Mat     = projSDP((x_Mat+x_Mat')/2);
x_star    = vec(x_Mat);
u_star    = (range_en_x-range_st_x)*rand(n_u,1);
x_reg     = x_star;
u_reg     = u_star;

x_0       = strange + (enrange-strange)*rand(n_x,1); 
u_0       = strange + (enrange-strange)*rand(n_u,1);
M         = strange/10 + 1/10*(enrange-strange)*rand(n_u,n_u);
M         = projSDP((M+M')/2);
Mvec      = vec(M);
mu_strcvx = 2*(norm(Mvec)+5);
%% When useuL_xu and usexL_xu is true then 
% strong convexity values larger than 9.75 when size is (d_x, d_u) = (1000,60) makes the algorithm non-convergent. Why? Where this value comes from?;
% strong convexity values smaller than 3.5 when size is (d_x, d_u) = (1000,60) makes the algorithm non-convergent. Why? Where this value comes from?;
%%
L_xx      = mu_strcvx;
L_uu      = mu_strcvx;
L_xu      = sqrt(n_x)*norm(M);
    function [fvalue ] = fxu(y,v)
       fvalue = mu_strcvx*norm(y-x_reg)^2-1/2*(y-x_reg)'*vec(M.*((v-u_reg)*(v-u_reg)'))-mu_strcvx/2*norm(v-u_reg)^2; 
    end
    function [Hessian_xx] = second_order_x(y,v)
       Hessian_xx = mu_strcvx*eye(n_x);
    end
    function [Hessian_uu] = second_order_u(y,v)
       Hessian_uu = -(M.*Mat(y-x_reg))-mu_strcvx*eye(n_u);
    end
    function [grad_x, grad_u, Hessian_xu ] = semi_second_order(y,v)
       grad_x     = mu_strcvx*(y-x_reg)-1/2*vec(M.*((v-u_reg)*(v-u_reg)'));
       grad_u     = -(M.*Mat(y-x_reg))*(v-u_reg) - mu_strcvx*(v-u_reg);
       Hessian_xu = repmat(-1/2*(M*(v-u_reg))',n_x,1);
    end
    function [y,v] = project(x,u)
        y         = max(x, strange);
        y         = min(y, enrange);
        %project on SDP
        ydif      = y-x_reg;
        x_Md      = Mat(ydif);
        x_Md      = projSDP((x_Md+x_Md')/2);
        y         = vec(x_Md) + x_reg;
        v         = max(u, strange);
        v         = min(v, enrange);
    end
    function [x_k, u_k] = prox_operator(L_xx, L_uu, L_xu, L_A_x, L_A_u, a_x, A_x, a_u, A_u, y, v, usexL_xu, useuL_xu)
        if usexL_xu && useuL_xu
           x_k      = y - (1/L_A_x)*a_x;
           u_k      = v - (1/L_A_u)*a_u; 
        elseif ~usexL_xu && useuL_xu
           x_k      = y - (A_x\a_x);
           u_k      = v - (1/L_A_u)*a_u;
        elseif usexL_xu && ~useuL_xu
           x_k      = y - (1/L_A_x)*a_x;
           u_k      = v - (A_u\a_u);
        else
           x_k      = y - (A_x\a_x);
           u_k      = v - (A_u\a_u);
        end
        [x_k,u_k]= project(x_k, u_k);
    end
    function [conv_meas] = compute_measure(meastype, x_k,u_k, y, v,iter)
        conv_meas  = norm(y-x_k)+norm(v-u_k);
    end
function_arguments.semi_second_order = @semi_second_order;
function_arguments.Hessian_x         = @second_order_x;
function_arguments.Hessian_u         = @second_order_u;
function_arguments.prox_operator     = @prox_operator;
function_arguments.objective_func    = @fxu;
function_arguments.converge_measure  = @compute_measure;

[y,v] = saddle_solver_ver2(options, x_0, u_0, function_arguments, n_x, n_u, L_xx, L_uu, L_xu, max_conv_meas, max_rel_gap, max_iter);
diff  = norm(y-x_star)+norm(v-u_star);
function [Mvec] = vec(M)
   Mvec = reshape(M,n_u^2,1);
end
    function [M] = Mat(Mvec)
        M = reshape(Mvec,n_u,n_u);
    end
end
