function [dualvars_k, x_k, solstatus ] = lssdp_ABCD_dual_noscale(max_conv_meas, max_rel_gap, max_iter, objectivefunc,x_0, dualvars_0, operators,learningparams,optparams)
%% This function solves the problem: min_{x\in \mathcal{C}} 1/2* \Vert x-x_0 \Vert^2
%       The set \mathcal{C} is defined by using operators. 
%       max_conv_meas is the maximum amount of etall if it exits
%       optimization loop sooner than max_iter. 
%       It will exit optimization loop sooner, if it feels there is no
%       progress which is determined if there is no enough change in last
%       noprogress_duration iterations. 
%       x_k : is the primal variable returned
%       dualvars_k: is the dual variable returned
%       There is no scaling in this function 
% Global variables 
global cnstData
%% setting parameters
dualsplitfunc = objectivefunc.dualsplit;
primal_func   = objectivefunc.primal;               % used just for computing primal value
rel_dist_func = objectivefunc.dist_x_opt;
A             = [operators.A_EC;operators.A_EV;operators.A_IC;operators.A_IV];
b_E           = [operators.b_EC;operators.b_EV];
s_I           = [operators.s_IC;operators.s_IV];
tol           = optparams.tol_ADMM;
maxit         = 100; %pcg maxiteration
soltype       = 3;   % compute y_E and y_I using 1: optimization using yalmip , 2: solve Ax=b using pcg ,3:solve Ax=b using cholesky factorization. 
sub_solvetype = 3;
tol           = 10^(-4);
restart_interval   = 1;
noprogres_duration = 10;
sum_change_etall   = 10^-4; % this must be dependent on max_conv_meas, but for now
%% setting initial values for improvement measures
Dobjective    = zeros(max_iter,1);
maxDobject    = realmin();
Pobjective    = zeros(max_iter,1);
etaIC         = zeros(max_iter,1);
etaCone       = zeros(max_iter,1);
etaEC         = zeros(max_iter,1);
etagap        = zeros(max_iter,1);
etaall        = zeros(max_iter,1);
dist_optimal  = zeros(max_iter,1);
%% initilizing loop
[y_ECtil,y_EVtil,y_ICtil,y_IVtil,Stil,Ztil,vtil] = load_dual_vars(dualvars_0);
Spre          = Stil; y_ECpre    = y_ECtil; y_EVpre    = y_EVtil; y_ICpre    = y_ICtil; y_IVpre    = y_IVtil;
iter          = 1;
converged     = false;
progressing   = true;
tk            = 1;
while ~converged && iter < max_iter && progressing
    %% Step 1
    [ y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I       ] = solve_dualsub_problems(sub_solvetype);    
    [x_k ]           = x_conv_from_dual_noproject(y_EC,y_EV,y_IC,y_IV,S,Z,v, x_0, operators);
    Dobjective(iter) = dualsplitfunc(y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
    if Dobjective(iter)> maxDobject
        maxDobject   = Dobjective(iter);
        maxy_EC      = y_EC; maxy_EV = y_EV; maxy_IC = y_IC; maxy_IV = y_IV; maxS = S; maxZ = Z; maxv = v;
    end
    %% Step 2: Update S, y_EC,y_EV,y_IC,y_IV,  
    [do_restart, resy_EC, resy_EV, resy_IC, resy_IV, resS, resZ , resv , res_tkplus] = restarting_analysis(iter, maxDobject, Dobjective);
    if ~do_restart
        tkplus           = (1+sqrt(1+4*tk^2))/2;
        betak            = (tk-1)/tkplus;
        y_ECtil          = y_EC + betak*(y_EC-y_ECpre);
        y_EVtil          = y_EV + betak*(y_EV-y_EVpre);
        y_ICtil          = y_IC + betak*(y_IC-y_ICpre);
        y_IVtil          = y_IV + betak*(y_IV-y_IVpre);
        Stil             = S    + betak*(S-Spre);
    else
        tkplus           = res_tkplus;
        y_ECtil  = resy_EC; y_EVtil  = resy_EV; y_ICtil  = resy_IC; y_IVtil  = resy_IV; Stil = resS; Ztil = resZ; vtil = resv;
    end
    y_ECpre = y_EC; y_EVpre = y_EV; y_ICpre = y_IC; y_IVpre = y_IV; Spre = S;
    tk           = tkplus;
    %% computing accuracy measures of the iteration
    Ay           = A'*[y_EC;y_EV;y_IC;y_IV];
    w_obetav     = x_0.w_obeta + cnstData.Hinv*(operators.B_EV'*y_EV+operators.B_IV'*y_IV);
    X            = proj_oncones(x_0.u+Ay+Z,cnstData.nSDP,cnstData.n_S,0);   
    Xp           = proj_oncones(x_0.u+Ay+S+Z,cnstData.nSDP,cnstData.n_S,0);   
    Y            = proj_onP(x_0.u+Ay+S,cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
    st           = min(x_0.st-[y_IC;y_IV],s_I);
    compConvMeasures();
    dist_optimal(iter) = rel_dist_func(x_conv_u(Xp,w_obetav,st));
    if etaall(iter) <= max_conv_meas && etagap(iter) <= max_rel_gap
       converged = true;
    end
    %% is it progressing?
    [sum_detail ] = sum_of_change_last_etall(etaall, iter, noprogres_duration );
    if sum_detail < sum_change_etall
        progressing = false; 
    end
    iter = iter + 1;
end
dualvars_k                 = dualvar_conv(y_EC, y_EV, y_IC, y_IV, S, Z ,  v );
x_k                        = x_conv_from_dual_fullproject(y_EC, y_EV, y_IC, y_IV, S, Z, v, x_0, operators);
solstatus.converged        = converged;
solstatus.progressing      = progressing;
solstatus.iter             = iter - 1;
solstatus.rel_gap          = etagap(iter-1);
solstatus.conv_meas        = etaall(iter-1);
solstatus.dist_to_optimal  = dist_optimal(iter-1);
solstatus.dualvar_valid    = true;
solstatus.primalvar_valid  = true;
return
    function compConvMeasures()
        Pobjective(iter)= primal_func(x_k, x_0, operators, learningparams, optparams);
        etaEC(iter)     = norm([operators.A_EC*X-operators.b_EC;operators.A_EV*X+operators.B_EV*w_obetav-operators.b_EV])/(1+norm([operators.b_EC;operators.b_EV]));
        etaIC(iter)     = norm(st+[-operators.A_IC*X;-operators.A_IV*X-operators.B_IV*w_obetav])/(1+norm(st));
        etaCone(iter)   = norm(X-Y)/(1+norm(X));
        %etagap is not computed based on my own formulation of the problem. 
        etagap(iter)    = abs(Dobjective(iter)-Pobjective(iter))/(1+abs(Pobjective(iter))+abs(Dobjective(iter)));
        etaall(iter)    = max ( max(etaEC(iter),etaCone(iter)),etaIC(iter));
    end
    function [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_dualsub_problems(typeofsolve)
             if typeofsolve == 1
                    [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_dualsub_problems();   
             elseif typeofsolve == 2    
                    [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCD2by3_solver_dualsub_problems();   
             elseif typeofsolve == 3
                    [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_Zvblocksolver();
             end
    end
    function [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_dualsub_problems
        star         = 1;
        Ay           = A'*[y_ECtil;y_EVtil;y_ICtil;y_IVtil];
        R            = Ay + Stil + x_0.u;
        [Z,v]        = projon_Conestar(cnstData.extendInd,R, x_0.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype,tol,maxit, ...
                                       y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype,tol,maxit,...
                                       y_EC, y_EV, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];                          
        S            = -(Ay+Z+x_0.u);
        S            = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype,tol,maxit,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_E    = norm(g_y_E);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype,tol,maxit,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_I    = norm(g_y_I);                         
    end 
    function [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCD2by3_solver_dualsub_problems()
        [Z,v]                       = solve_BCD_Zvblock();
        [y_EC, y_EV, y_IC, y_IV, S] = solve_BCD_SyIyEblock_solver(Z,v);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_I    = norm(g_y_I); normg_y_E    = norm(g_y_E);                         
    end
    function [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_Zvblocksolver()
        star         = 1;
        Ay           = A'*[y_ECtil;y_EVtil;y_ICtil;y_IVtil];
        R            = Ay + Stil + x_0.u;
        [Z,v]        = projon_Conestar(cnstData.extendInd,R, x_0.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
        %[Z1,v1]                       = solve_BCD_Zvblock();
%         diffnorm     = norm(Z-Z1)+norm(v-v1);
        [y_EC, y_EV, y_IC, y_IV, S] = solve_BCDDetail_SyIyEblock_problems(Z,v);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_I    = norm(g_y_I); normg_y_E    = norm(g_y_E);                         
    end
    function [y_EC, y_EV, y_IC, y_IV, S, normg_y_E, normg_y_I ]        = solve_BCDDetail_SyIyEblock_problems(Z, v)
        star         = 1;
        Ay           = A'*[y_ECtil;y_EVtil;y_ICtil;y_IVtil];
        R            = Ay + Stil + x_0.u;
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype,tol,maxit, ...
                                       y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype,tol,maxit,...
                                       y_EC, y_EV, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];                          
        S            = -(Ay+Z+x_0.u);
        S            = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype,tol,maxit,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_E    = norm(g_y_E);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype,tol,maxit,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_I    = norm(g_y_I);                         
    end
    function [y_EC, y_EV, y_IC, y_IV, S]                               = solve_BCD_SyIyEblock_solver(Z,v) 
        y_ECyal      = sdpvar(operators.n_AEC,1);
        y_EVyal      = sdpvar(operators.n_AEV,1);
        y_ICyal      = sdpvar(operators.n_AIC,1);
        y_IVyal      = sdpvar(operators.n_AIV,1);
        S_mat        = sdpvar(cnstData.nSDP,cnstData.nSDP);
        p_NN_dual    = sdpvar(cnstData.n_S,1);
        Syal         = [reshape(S_mat,cnstData.nSDP*cnstData.nSDP,1);p_NN_dual];
        dualobjfunc  = objectivefunc.dual;
        dualvars     = dualvar_conv(y_ECyal, y_EVyal, y_ICyal, y_IVyal, Syal, Z ,  v );
        cObjective   = -dualobjfunc(dualvars, x_0, operators, learningparams, optparams);
        dcConstraint = [ S_mat>=0, p_NN_dual>=0];
        sol          = optimize(dcConstraint, cObjective);
        if sol.problem==0 
           obj_val   = -value(cObjective);
           y_EC      =  value(y_ECyal); 
           y_EV      =  value(y_EVyal);
           y_IC      =  value(y_ICyal); 
           y_IV      =  value(y_IVyal);
           S         =  value(Syal);
        else 
            assert(true,'Error cannot solve problem for y_E in routine proxLmu_y_E');
        end 
    end
    function [Z,v]                                                     = solve_BCD_Zvblock()
        vyal          = sdpvar(operators.n_AIC+operators.n_AIV,1);
        q_NN_dual     = sdpvar(cnstData.n_u,1);
        last_col_dual = [zeros(cnstData.n_S,1);q_NN_dual];
        Z_NN_Mat      = [zeros(cnstData.nSDP-1,cnstData.nSDP-1),last_col_dual;last_col_dual',0];
        Z_NNyal       = [reshape(Z_NN_Mat,cnstData.nSDP*cnstData.nSDP,1);zeros(cnstData.n_S,1)]; 
        dualobjfunc   = objectivefunc.dual;
        dualvars      = dualvar_conv(y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z_NNyal ,  vyal );
        cObjective    = -dualobjfunc(dualvars, x_0, operators, learningparams, optparams);
        dcConstraint  = [ q_NN_dual>=0, vyal>=0];
        sol           = optimize(dcConstraint, cObjective);
        if sol.problem==0 
           obj_val    = -value(cObjective);
           Z          = value(Z_NNyal);
           v          = value(vyal);
        else 
            assert(true,'Error cannot solve problem for y_E in routine proxLmu_y_E');
        end
    end
    function [do_restart, resy_EC, resy_EV, resy_IC, resy_IV, resS, resZ , resv , res_tkplus ] = restarting_analysis(iter, maxDobject, Dobjective)
        do_restart = false;
        resy_EC= 0; resy_EV=0; resy_IC=0; resy_IV=0; resS=0; resZ=0; resv=0; res_tkplus=0;
        if iter==1, return,end
        if mod(iter, restart_interval)~=0,  return , end
        restart_method= 3;
        if restart_method == 1
           if maxDobject > Dobjective(iter)
               do_restart = true;  
               resy_EC    = maxy_EC; resy_EV = maxy_EV; resy_IC = maxy_IC; resy_IV = maxy_IV; resS = maxS; resZ = maxZ; resv = maxv;
               res_tkplus = 2*tk; % 1; all of it may be bad.  
               return ;
           end
        elseif restart_method == 2 % it will blocked in a loop 
            if maxDobject > Dobjective(iter)
               do_restart = true;  
               resy_EC    = y_ECtil; resy_EV = y_EVtil; resy_IC = y_ICtil; resy_IV = y_IVtil; resS = Stil; resZ = Ztil; resv = vtil;
               res_tkplus = 1;%2*tk;%tk;%1; all of it may be bad 
               return ;
            end
        elseif restart_method == 3 % this is the best method so far. but it can only reach to 0.0981 relative distance to optimal. 
                                   % I think this is because solving BCD
                                   % with detail is not accurate and it is
                                   % the amount of error in it which
                                   % determines how much we can reach close
                                   % to optimal. 
            if Dobjective(iter-1) > Dobjective(iter)
               do_restart = true;  
               resy_EC    = y_ECtil; resy_EV = y_EVtil; resy_IC = y_ICtil; resy_IV = y_IVtil; resS = Stil; resZ = Ztil; resv = vtil;
               res_tkplus = 1; 
               return ;
            end
        end
    end
    function [sum_detail ]                                             = sum_of_change_last_etall(etaall, iter, duration )
       if iter <= duration, 
           sum_detail = realmax; 
           return;
       end
       start_iter = iter-duration + 1;
       sum_detail = sum(abs(etaall(start_iter-1:iter-1)-etaall(start_iter:iter)));
    end
end
