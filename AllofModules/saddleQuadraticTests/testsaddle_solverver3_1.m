function testsaddle_solverver3_1 

n_u       = 100; 
n_x       = n_u^2;
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
max_conv_meas = 10^-4;
max_rel_gap   = 10^-3;
range_st_x= 0.25;
range_en_x= 0.75;
x_reg    = (range_en_x-range_st_x)*rand(n_x,1);
x_Mat     = Mat(x_reg);
x_Mat     = projSDP((x_Mat+x_Mat')/2);
x_reg    = vec(x_Mat);
u_reg    = (range_en_x-range_st_x)*rand(n_u,1);

x_0       = strange + (enrange-strange)*rand(n_x,1); 
u_0       = strange + (enrange-strange)*rand(n_u,1);
M         = strange/10 + 1/10*(enrange-strange)*rand(n_u,n_u);
M         = projSDP((M+M')/2);
Mvec      = vec(M);
q         = (norm(Mvec)+1);
mu_strcvx_x = q ; % n(n+1)/2/n 
mu_strcvx_u = q *n_u;
do_search = false;
    function [x_star, u_star] = search_optimal()
        delta = 0.1;
        min_grad = realmax();
        for i=-1:delta:1
           for j = -1:delta:1
              for x_1 = 0:delta:1
                 for x_4 = 0:delta:1
                    for x_3 = -1:delta:1 
                         y     = x_project([x_1;x_3;x_3;x_4],[i;j]); v = u_project([x_1;x_3;x_3;x_4],[i;j]);
                        [grad_x, grad_u, Hessian_xu ] = semi_second_order(y,v);
                        g = norm(grad_x)+norm(grad_u);
                        if g < min_grad
                            min_grad = g;
                            x_star   = [x_1;x_3;x_3;x_4];
                            u_star   = [i;j];
                        end
                    end
                 end
              end
           end
        end        
    end
if do_search
    [x_star, u_star] = search_optimal();
    save('optimal_search','x_reg','u_reg','M','x_star','u_star','mu_strcvx');
% else
%     load('optimal_search','x_reg','u_reg','M','x_star','u_star','mu_strcvx');
end

%% When useuL_xu and usexL_xu is true then 
% strong convexity values larger than 9.75 when size is (d_x, d_u) = (1000,60) makes the algorithm non-convergent. Why? Where this value comes from?;
% strong convexity values smaller than 3.5 when size is (d_x, d_u) = (1000,60) makes the algorithm non-convergent. Why? Where this value comes from?;
%%
L_xx      = mu_strcvx_x;
L_uu      = mu_strcvx_u;
L_xu      = 1/2*n_u*sqrt(n_u)*norm(M);
    function [fvalue ] = fxu(y,v)
       fvalue = mu_strcvx_x*norm(y-x_reg)^2-1/2*(y-x_reg)'*vec(M.*((v-u_reg)*(v-u_reg)'))-mu_strcvx_u/2*norm(v-u_reg)^2 ;
    end
    function [Hessian_xx] = second_order_x(y,v)
       Hessian_xx = mu_strcvx_x*eye(n_x);
    end
    function [Hessian_uu] = second_order_u(y,v)
       Hessian_uu = -(M.*Mat(y))-mu_strcvx_u*eye(n_u);
    end
    function [grad_x, grad_u, Hessian_xu ] = semi_second_order(y,v)
       grad_x     = mu_strcvx_x*(y-x_reg)-1/2*vec(M.*((v-u_reg)*(v-u_reg)'));
       grad_u     = -(M.*Mat(y-x_reg))*(v-u_reg) - mu_strcvx_u*(v-u_reg);
       Hessian_xu = repmat(-1/2*(M*(v-u_reg))',n_x,1);
    end
    function [y ] = x_project(x,u)
        y         = max(x, strange);
        y         = min(y, enrange);
        %project on SDP
%         x_M       = Mat(y);
%         x_M       = projSDP((x_M+x_M')/2);
%         y         = vec(x_M);
        ydif      = y-x_reg;
        x_Md      = Mat(ydif);
        x_Md      = projSDP((x_Md+x_Md')/2);
        y         = vec(x_Md) + x_reg;
    end
    function [v ] = u_project(x,u)
        v         = max(u, strange);
        v         = min(v, enrange);
    end
    function [x_k] = x_prox_operator(L_xx, L_xu, L_A_x, a_x, A_x, y, v, usexL_xu)
        if usexL_xu 
           x_k      = y - (1/L_A_x)*a_x;
        else
           x_k      = y - (A_x\a_x);
        end
        [x_k]= x_project(x_k, v);
    end
    function [u_k] = u_prox_operator(L_uu, L_xu, L_A_u, a_u, A_u, y, v, useuL_xu)
        if useuL_xu
           u_k      = v - (1/L_A_u)*a_u; 
        else
            u_k      = v - (A_u\a_u);
        end
        [u_k]= u_project(y, u_k);
    end
    function [conv_meas] = compute_measure(meastype, x_k,u_k, y, v,iter)
        conv_meas  = norm(y-x_k)+norm(v-u_k);
    end
function_arguments.semi_second_order = @semi_second_order;
function_arguments.Hessian_x         = @second_order_x;
function_arguments.Hessian_u         = @second_order_u;
function_arguments.x_prox_operator   = @x_prox_operator;
function_arguments.u_prox_operator   = @u_prox_operator;
function_arguments.objective_func    = @fxu;
function_arguments.converge_measure  = @compute_measure;

[y,v] = saddle_solver_ver4(options, x_0, u_0, function_arguments, n_x, n_u, L_xx, L_uu, L_xu, max_conv_meas, max_rel_gap, max_iter);
df    = norm(y-x_reg)+norm(v-u_reg);
[grad_x, grad_u, Hessian_xu ] = semi_second_order(y,v);
g_ret = norm(grad_x)+norm(grad_u);
[grad_x, grad_u, Hessian_xu ] = semi_second_order(x_reg,u_reg);
g_reg = norm(grad_x)+norm(grad_u);


function [Mvec] = vec(M)
   Mvec = reshape(M,n_u^2,1);
end
    function [M] = Mat(Mvec)
        M = reshape(Mvec,n_u,n_u);
    end
end
