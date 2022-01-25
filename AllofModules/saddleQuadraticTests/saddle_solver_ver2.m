function [y,v, solstatus] = saddle_solver_ver2(options, x_0, u_0, function_arguments, n_x, n_u, L_xx, L_uu, L_xu, max_conv_meas, max_rel_gap, max_iter)
usexHessian = options.usexHessian;
useuHessian = options.useuHessian;
useuL_xu    = options.useuL_xu;
usexL_xu    = options.usexL_xu;
usexpcg     = options.usexpcg;
useupcg     = options.useupcg;
pcgtolstart = options.pcgtolstart;
pcgtolend   = options.pcgtolend;

semi_second_order = function_arguments.semi_second_order;
if usexHessian 
   get_Hessian_xx    = function_arguments.Hessian_x;
end
if useuHessian
   get_Hessian_uu    = function_arguments.Hessian_u;
end
proximal_operator = function_arguments.prox_operator;
objectivefunc     = function_arguments.objective_func;
measurefunc       = function_arguments.converge_measure;
L_A_u             = (L_xu^2)/L_xx + L_uu;
L_A_x             = (L_xu^2)/L_uu + L_xx;
A_u               = 1;
A_x               = 1;
objective   = zeros(max_iter,1);
etaall      = zeros(max_iter,1);
gap         = zeros(max_iter,1);
%% Setup restart and no progress mechanisms. Also, measuretype
restart_interval   = 1;     % check restarting every restart_interval iterations
noprogres_duration = 10;    % check if no progress has been made in the last noprogres_duration iterations
sum_change_etall   = max_conv_meas; % this must be dependent on max_conv_meas, but for now
progressing        = true;
meastype           = 1;
x_avg     = x_0;
u_avg     = u_0;
x_avgpre  = x_avg;
u_avgpre  = u_avg;

y         = x_0; 
v         = u_0;
x_kpre    = x_0;
u_kpre    = u_0;
mingap    = realmax;
tk        = 1;
iter      = 1;
converged = false;
while ~converged && (iter <= max_iter)
    [grad_x, grad_u, Hessian_xu ] = semi_second_order(y,v); 
    if ~usexHessian && ~useuHessian 
       if ~useuL_xu 
          A_u      = Hessian_xu'*1/L_xx*Hessian_xu + L_uu*eye(n_u);
       end
       a_u      = -(grad_u-Hessian_xu'*(1/L_xx)*grad_x);  
       if ~usexL_xu 
          A_x      = +Hessian_xu *1/L_uu*Hessian_xu'+ L_xx*eye(n_x);
       end
       a_x      = grad_x+Hessian_xu*(1/L_uu)*grad_u; 
       [x_k, u_k]      = proximal_operator(L_xx, L_uu, L_xu, L_A_x, L_A_u, a_x, A_x, a_u, A_u, y, v, usexL_xu, useuL_xu);
    elseif usexHessian && ~useuHessian
       Hessian_xx = get_Hessian_xx(y,v); 
       if ~usexL_xu 
          A_x      = +Hessian_xu *1/L_uu*Hessian_xu'+ Hessian_xx;%L_xx*eye(n_x);
       end
       a_x      = grad_x+Hessian_xu*(1/L_uu)*grad_u;          
       M_xx_inv = inv(Hessian_xx); 
       a_u      = -(grad_u-Hessian_xu'*M_xx_inv*grad_x);
       [x_k, u_k]      = proximal_operator(L_xx, L_uu, L_xu, L_A_x, L_A_u, a_x, A_x, a_u, A_u, y, v, usexL_xu, useuL_xu);
    elseif ~usexHessian && useuHessian
       Hessian_uu = get_Hessian_uu(y,v); 
       if ~useuL_xu 
          A_u      = Hessian_xu'*1/L_xx*Hessian_xu + Hessian_uu;%L_uu*eye(n_u);
       end
       a_u      = -(grad_u-Hessian_xu'*(1/L_xx)*grad_x);
       a_x      = grad_x-Hessian_xu*(Hessian_uu\grad_u);
       [x_k, u_k]      = proximal_operator(L_xx, L_uu, L_xu, L_A_x, L_A_u, a_x, A_x, a_u, A_u, y, v, usexL_xu, useuL_xu);
    else %% usexHessian && useuHessian
       Hessian_xx = get_Hessian_xx(y,v); 
       M_xx_inv = inv(Hessian_xx); 
       A_u      = Hessian_xu'*M_xx_inv*Hessian_xu + L_uu*eye(n_u);
       a_u      = -(grad_u-Hessian_xu'*M_xx_inv*grad_x); 
       Hessian_uu = get_Hessian_uu(y,v);
       M_uu_inv = inv(Hessian_uu);
       A_x      = -Hessian_xu *M_uu_inv*Hessian_xu'+ L_xx*eye(n_x);
       a_x      = grad_x-Hessian_xu*M_uu_inv*grad_u; 
       [x_k, u_k]      = proximal_operator(L_xx, L_uu, L_xu, L_A_x, L_A_u, a_x, A_x, a_u, A_u, y, v, usexL_xu, useuL_xu);
    end  
    x_avgpre  = x_avg;
    u_avgpre  = u_avg;
    x_avg     = (x_avg*(iter-1)+ x_k)/iter;
    u_avg     = (u_avg*(iter-1)+ u_k)/iter;    
    objective(iter) = objectivefunc(x_avg, u_avg);
    preobj          = objectivefunc(y,v);
    max_u_obj       = objectivefunc(y,   u_k);
    min_x_obj       = objectivefunc(x_k, v);
    gap(iter)       = abs(max_u_obj-min_x_obj);% max_u_obj must be greater than min_x_obj except for the first iteration. 
    if gap(iter) < mingap, mingap = gap(iter); end
    
    etaall(iter)  = measurefunc(meastype, x_avg, u_avg, x_avgpre, u_avgpre, iter);
    if etaall(iter) <= max_conv_meas && gap(iter) <= max_rel_gap
        converged = true;
    end
    tkplus   = (1+sqrt(1+4*tk^2))/2;
    betak    = (tk-1)/tkplus;
    [do_restart, res_tkplus, x_krestart, u_krestart] = restarting_analysis(iter, mingap, gap);
    if do_restart 
        tk       = res_tkplus;
        y        = x_krestart;
        v        = u_krestart;
    else
        y        = x_k + betak*(x_k-x_kpre);
        v        = u_k + betak*(u_k-u_kpre);
        x_kpre   = x_k;
        u_kpre   = u_k;
        tk       = tkplus;
    end
    tk = 1;
    [sum_detail ] = sum_of_change_last_etall(etaall, iter, noprogres_duration );
    if sum_detail < sum_change_etall
        progressing = false; 
    end
    iter     = iter + 1;
end
solstatus.converged        = converged;
solstatus.progressing      = progressing;
solstatus.iter             = iter - 1;
solstatus.rel_gap          = gap(iter-1);
solstatus.conv_meas        = etaall(iter-1);
solstatus.dist_to_optimal  = realmax;
solstatus.dualvar_valid    = true;
solstatus.primalvar_valid  = true;
return
    function [do_restart, res_tkplus, x_krestart, u_krestart] = restarting_analysis(iter, mingap, gap)
        do_restart = false;
        x_krestart = 0; u_krestart = 0;res_tkplus = 0;
        restart_method  = 3;
        if iter==1, return,end
        if mod(iter, restart_interval)~=0,  return , end
        if restart_method == 3
            if gap(iter-1) < gap(iter)
                do_restart = true;
                x_krestart = x_kpre;
                u_krestart = u_kpre;
                res_tkplus = 1;
                return;
            end
        end
    end
    function [sum_detail ]                                    = sum_of_change_last_etall(etaall, iter, duration )
       if iter <= duration, 
           sum_detail = realmax; 
           return;
       end
       start_iter = iter-duration + 1;
       sum_detail = sum(abs(etaall(start_iter-1:iter-1)-etaall(start_iter:iter)));
    end
end