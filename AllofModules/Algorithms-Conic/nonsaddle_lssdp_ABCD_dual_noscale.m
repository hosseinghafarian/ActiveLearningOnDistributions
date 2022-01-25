function [alpha_k, x_k, dualvars_k, solstatus ] = nonsaddle_lssdp_ABCD_dual_noscale(x_G, alpha_0, dualvars_0, operators,optparams,learningparams, progress_func,verbose, max_conv_meas, max_rel_gap, max_iter)
% Global variables 
global cnstData
global optimals
%% Setting objective function components 
objectivefunc.primal      = @primal_objective;
objectivefunc.dual        = @  dual_objective_func;
objectivefunc.dualsplit   = @  dual_objective_split;
objectivefunc.LHSRHS      = @  dual_objective_split_LHS_RHS;
objectivefunc.update_LHS_Mat_y_E_y_I = @ update_Qinv_LHS_Mat_y_E_y_I;
objectivefunc.dist_x_opt  = @relative_dist_to_x_opt;
objectivefunc.ProofChecker= @NestCompProofChecker; 
%% setting initial values for improvement measures
Dobjective                = zeros(max_iter,1);
maxDobject                = realmin();
Pobjective                = zeros(max_iter,1);
etaIC                     = zeros(max_iter,1);
etaCone                   = zeros(max_iter,1);
etaEC                     = zeros(max_iter,1);
etagap                    = zeros(max_iter,1);
etalpha                   = zeros(max_iter,1);
etaall                    = zeros(max_iter,1);
dist_optimal              = zeros(max_iter,1);
%% Setup for Solving y_E and y_I
% Set up LHS and Cholesky factorization for computing y_E and y_I in iterations 
dualsplitfunc = objectivefunc.dualsplit;
primal_func   = objectivefunc.primal;               % used just for computing primal value
update_LHS_Mat_y_E_y_I = objectivefunc.update_LHS_Mat_y_E_y_I;
[operators.LHS_y_E, operators.LHS_y_I, operators.CHOL_H_y_E, operators.CHOL_H_y_I] ...
                = update_LHS_Mat_y_E_y_I(operators);
A               = [operators.A_EC;operators.A_EV;operators.A_IC;operators.A_IV];
b_E             = [operators.b_EC;operators.b_EV];
s_I             = [operators.s_IC;operators.s_IV];
% Initialize Parameters for solving y_E and y_I
tol4y_E_y_I     = optparams.tol_ADMM;
maxit4y_E_y_I   = 100; %pcg maxiteration
soltype4y_E_y_I = 3;   % compute y_E and y_I using 1: optimization using yalmip , 2: solve Ax=b using pcg ,3:solve Ax=b using cholesky factorization. 
%% Setup restart, no progress mechanisms
sub_solvetype      = 1;
restart_interval   = 1;     % check restarting every restart_interval iterations
noprogres_duration = 10;    % check if no progress has been made in the last noprogres_duration iterations
sum_change_etall   = 10^-4; % this must be dependent on max_conv_meas, but for now
%% initilizing loop
% fetch starting values for x, dualvars and alpha
[x_0 ]        = update_x_0_alpha(alpha_0, x_G);
norm_x_0      = x_norm(x_0,cnstData.Q);
alpha_ktil    = alpha_0;
[y_ECtil,y_EVtil,y_ICtil,y_IVtil,Stil,Ztil,vtil] = load_dual_vars(dualvars_0);
% set starting  and previous values for variables 
alpha_k       = alpha_ktil;
alpha_pre     = alpha_ktil;
Spre          = Stil; y_ECpre    = y_ECtil; y_EVpre    = y_EVtil; y_ICpre    = y_ICtil; y_IVpre    = y_IVtil;
% set starting loop values 
iter          = 1;
converged     = false;
progressing   = true;
tk            = 1;
while ~converged && iter < max_iter && progressing
    %% Step 1
    [ y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I, alpha_k   ] = solve_dualsub_problems(sub_solvetype);    
    [x_k ]           = x_conv_from_dual_fullproject(y_EC,y_EV,y_IC,y_IV,S,Z,v, x_0, operators);

    Dobjective(iter) = dualsplitfunc(y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams, norm_x_0);
    if Dobjective(iter)> maxDobject
        maxDobject   = Dobjective(iter);
        maxy_EC      = y_EC; maxy_EV = y_EV; maxy_IC = y_IC; maxy_IV = y_IV; maxS = S; maxZ = Z; maxv = v;
    end
    %% Step 2: Update S, y_EC,y_EV,y_IC,y_IV,  
    [do_restart, resy_EC, resy_EV, resy_IC, resy_IV, resS, resZ , resv , res_tkplus] = restarting_analysis(iter, maxDobject, Dobjective);
    if ~do_restart
        tkplus       = (1+sqrt(1+4*tk^2))/2;
        betak        = (tk-1)/tkplus;
        y_ECtil      = y_EC + betak*(y_EC-y_ECpre);
        y_EVtil      = y_EV + betak*(y_EV-y_EVpre);
        y_ICtil      = y_IC + betak*(y_IC-y_ICpre);
        y_IVtil      = y_IV + betak*(y_IV-y_IVpre);
        Stil         = S    + betak*(S-Spre);
        alpha_ktil   = alpha_k;% + betak*(alpha_k-alpha_pre);
    else
        tkplus           = res_tkplus;
        y_ECtil  = resy_EC; y_EVtil  = resy_EV; y_ICtil  = resy_IC; y_IVtil  = resy_IV; Stil = resS; Ztil = resZ; vtil = resv;
    end
    y_ECpre = y_EC; y_EVpre = y_EV; y_ICpre = y_IC; y_IVpre = y_IV; Spre = S; alpha_pre = alpha_k;
    tk           = tkplus;
    %% computing accuracy measures of the iteration
    Ay           = A'*[y_EC;y_EV;y_IC;y_IV];
    w_obetav     = x_0.w_obeta + cnstData.Hinv*(operators.B_EV'*y_EV+operators.B_IV'*y_IV);
    X            = proj_oncones(x_0.u+Ay+Z,cnstData.nSDP,cnstData.n_S,0);   
    Xp           = proj_oncones(x_0.u+Ay+S+Z,cnstData.nSDP,cnstData.n_S,0);   
    Y            = proj_onP(x_0.u+Ay+S,cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
    st           = min(x_0.st-[y_IC;y_IV],s_I);
    compConvMeasures();
    %dist_optimal(iter) = rel_dist_func(x_conv_u(Xp,w_obetav,st));
    if etaall(iter) <= max_conv_meas && etagap(iter) <= max_rel_gap
       converged = true;
    end
    %% is it progressing?
    [sum_detail ] = sum_of_change_last_etall(etaall, iter, noprogres_duration );
    if sum_detail < sum_change_etall
        progressing = false; 
    end
    % update x_0 whenever alpha_k changes
    [x_0 ] = update_x_0_alpha(alpha_k,x_G);
    iter = iter + 1;
end
%% setup return values for alpha_k, x_k and dualvars_k
dualvars_k                 = dualvar_conv(y_EC, y_EV, y_IC, y_IV, S, Z ,  v );
x_k                        = x_conv_from_dual_fullproject(y_EC, y_EV, y_IC, y_IV, S, Z, v, x_0, operators);
%% setup status values for problem 
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
        etalpha(iter)   = norm(alpha_ktil-alpha_pre)/(1+norm(alpha_k));
        etaall(iter)    = max(max ( max(etaEC(iter),etaCone(iter)),etaIC(iter)),etalpha(iter));
    end
    function [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I, alpha_knext ] = solve_dualsub_problems(typeofsolve)

             if typeofsolve == 1
                    [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_dualsub_problems();   
             elseif typeofsolve == 2    
                    [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCD2by3_solver_dualsub_problems();   
             elseif typeofsolve == 3
                    [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_Zvblocksolver();
             end
             [x_k ]           = x_conv_from_dual_fullproject(y_EC,y_EV,y_IC,y_IV,S,Z,v, x_0, operators);
             [alpha_knext]    = proxf_alpha(learningparams, optparams, alpha_0, x_k);
             [alpha_dual]     = dual_alpha_method(learningparams, optparams, alpha_k, alpha_0, x_k);
             tau_k            = 0.7;
             alpha_knext      = tau_k* alpha_knext + (1-tau_k)* alpha_dual;
             [f,g_x,g_alpha]  = f_xAlpha_grad(x_k,alpha_ktil,learningparams);
    end
    function [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_dualsub_problems
        star         = 1;
        Ay           = A'*[y_ECtil;y_EVtil;y_ICtil;y_IVtil];
        R            = Ay + Stil + x_0.u;
        [Z,v]        = projon_Conestar(cnstData.extendInd,R, x_0.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I, ...
                                       y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I,...
                                       y_EC, y_EV, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];                          
        S            = -(Ay+Z+x_0.u);
        S            = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_E    = norm(g_y_E);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I,...
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
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I, ...
                                       y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I,...
                                       y_EC, y_EV, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];                          
        S            = -(Ay+Z+x_0.u);
        S            = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_E    = norm(g_y_E);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,soltype4y_E_y_I,tol4y_E_y_I,maxit4y_E_y_I,...
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
    function [x_0 ] = update_x_0_alpha(alpha_k,x_G)
        % This function computes x_0, whenever alpha changes. 
        [c_k]           = M_of_alpha(alpha_k,learningparams);
        g_acc_x.u       = c_k;
        g_acc_x.w_obeta = zeros(cnstData.n_S,1);
        g_acc_x.st      = 0;
        beta_kx         = learningparams.rhox;
        x_0.u           = (1/beta_kx)*(- g_acc_x.u                     + beta_kx*x_G.u      );
        x_0.w_obeta     = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_G.w_obeta);
        x_0.st          = (1/beta_kx)*(- g_acc_x.st                    + beta_kx*x_G.st     ); 
    end
    function [sum_detail ]                                             = sum_of_change_last_etall(etaall, iter, duration )
       if iter <= duration, 
           sum_detail = realmax; 
           return;
       end
       start_iter = iter-duration + 1;
       sum_detail = sum(abs(etaall(start_iter-1:iter-1)-etaall(start_iter:iter)));
    end
    function [alpha_new]  = dual_alpha_method(learningparams, optparams, alpha_k, alpha_0, x_k)
        % How to consider effect of normalization with respect to x_0 which
        % contains alpha? it seems that based on the function update_x_0_alpha, we must divide
        % objective function by 1/learningparams.rhox. But this is
        % incorrect, since alpha updates in the saddle point and in that
        % scaling doesnot affect the optimal point. 
        graditer  = 1000;
        tol_grad  = 10^-5;
        D_x       = 0.4;%cnstData.nConic*sqrt(2);
        normxk    = norm(x_k.u); % approximately:must change
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        rhop      = learningparams.rhoalpha;
        h_k       = 1/(optparams.L_alpha);
        g_acc_x.w_obeta  = learningparams.lambda_o*cnstData.K*x_k.w_obeta;
        g_acc_x.st       = 0;
        norm_gacc_beta   = norm(g_acc_x.w_obeta)^2;
        [l_of_x, G_of_x] = x_obj_get_lG_of_x(x_k);
        KG               = (cnstData.KE.* G_of_x(1:nap,1:nap)) / learningparams.lambda;
        KGwithoutx       =  cnstData.KE / learningparams.lambda;
        L_a_local        = norm(KG)+rhop+ D_x*normxk*norm(KGwithoutx); %% Attention: Lipschitz constant must be correct for this function. 
        h_of_x           = l_of_x;
        function [gout ] = grad_alpha(alpha_new)
            [c_k]           = M_of_alpha(alpha_new,learningparams);
            sum_c_k         = sum(c_k);
            l_star          = [ones(n_S,1);zeros(nap-n_S,1)];
            diag_lKGwithoutx= l_star - KGwithoutx* alpha_new;
            norm_g_acc_f    = sqrt(norm(c_k)^2+norm_gacc_beta); 
            dualadjustterm  = D_x* sum_c_k/norm_g_acc_f*diag_lKGwithoutx;
            gout      = -h_of_x + KG*alpha_new + rhop*(alpha_new-alpha_0) + dualadjustterm;
        end
        function [alpha_prj ] = project_alpha(alpha_new)
            alpha_new = max(alpha_new, cnstData.lo);
            alpha_prj = min(alpha_new, cnstData.up);
        end
        %[alpha_new] = accelerated_projected_gradient(alpha_k, @grad_alpha, @project_alpha, L_a_local, rhop, graditer, tol_grad);
        [alpha_new ] = projected_gradient_descent(alpha_k, @grad_alpha, @project_alpha, L_a_local, graditer, tol_grad);
    end
    function [alpha_new]  = proxf_alpha(learningparams, optparams, alpha0, x_k)
        % it is very interesting that a few iterations of accelerated_first
        % order method is good ( 5 iterations) and is better than first
        % order method, when rhop, strong convexity parameter with respect
        % to alpha is small. But if number of iterations of
        % acclerated_first order method increases, in terms of the number
        % of iterations of ActiveOutlierSaddle15 it is bad and in
        % particular worse than simple gradient scheme with a single
        % iteration. Also, accelerated method with a few iterations is also
        % better than second order method, which is very much slower than
        % it. Even in the simple gradient method, if we increase the number
        % of steps we will not always get a worth the effort result. 
        % Conclusion: it seems that in the simple stages of the algorithm
        % if we solve the subproblem regarding alpha accurately, we make
        % the problem harder to solve. But solving it a little bit more
        % accurate, helps to go a better route in terms of the variable x.
        % so, It seems that it is better to start with a non-exact
        % algorithm for the subproblem and the increasingly, increase the
        % level of accurcy in the latter itreations. 
        second_order_alpha      = false;
        accelerated_first_alpha = true;
        if second_order_alpha 
            [alpha_new]  = sec_order_proxf_alpha(learningparams, optparams, alpha0, x_k);
        elseif accelerated_first_alpha
            [alpha_new]  = accelerated_first_order_proxf_alpha(learningparams, optparams, alpha0, x_k);
        else    
            [alpha_new]  = first_order_proxf_alpha(learningparams, optparams, alpha0, x_k); 
        end
    end
    function [alpha_new]  = first_order_proxf_alpha(learningparams, optparams, alpha0, x_k)
        graditer  = 5;
        tol_grad   = 10^-5;
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        rhop      = learningparams.rhoalpha;
        h_k       = 1/(optparams.L_alpha);
        [l_of_x, G_of_x] = x_obj_get_lG_of_x(x_k);
        KG        = (cnstData.KE.* G_of_x(1:nap,1:nap)) / learningparams.lambda;
        L_a_local = norm(KG)+rhop;
        h_of_x    = l_of_x;
        function [gout ] = grad_alpha(alpha_new)
            gout      = -h_of_x + KG*alpha_new + rhop*(alpha_new-alpha0);
        end
        function [alpha_prj ] = project_alpha(alpha_new)
            alpha_new = max(alpha_new, cnstData.lo);
            alpha_prj = min(alpha_new, cnstData.up);
        end
        
        [alpha_new ] = projected_gradient_descent(alpha_k, @grad_alpha, @project_alpha, L_a_local, graditer, tol_grad);
    end
    function [alpha_new]  = accelerated_first_order_proxf_alpha(learningparams, optparams, alpha0, x_k)
        acciter   = 5;
        tol_grad  = 10^-5;
        nap       = cnstData.nap;
        rhop      = learningparams.rhoalpha;
        [l_of_x, G_of_x] = x_obj_get_lG_of_x(x_k);
        KG        = (cnstData.KE.* G_of_x(1:nap,1:nap)) / learningparams.lambda;
        L_a_local = norm(KG)+rhop;
        h_of_x    = l_of_x;
        function [gout ] = grad_alpha(alpha_new)
            gout      = -h_of_x + KG*alpha_new + rhop*(alpha_new-alpha0);
        end
        function [alpha_prj ] = project_alpha(alpha_new)
            alpha_new = max(alpha_new, cnstData.lo);
            alpha_prj = min(alpha_new, cnstData.up);
        end
        [alpha_new] = accelerated_projected_gradient(alpha_k, @grad_alpha, @project_alpha, L_a_local, rhop, acciter, tol_grad);
    end
    function [alpha_new]  = sec_order_proxf_alpha(learningparams, optparams, alpha0, x_k)
        global KG;
        global h_of_x;
        global alphapref;
        global rhop;
        function [fout,gout]                   = f_lG_x_alpha(alphav)
            %global h_of_x;global alphapref;global rhop;
            fout  = -alphav'*h_of_x + 1/2* alphav'*KG*alphav ;%+ rhop/(2)*norm(alphav-alphapref)^2;
            gout  = -        h_of_x +              KG*alphav ;%+ rhop*(alphav-alphapref);
        end
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        [l_of_x, G_of_x] = x_obj_get_lG_of_x(x_k);
        rhop      = learningparams.rhoalpha;
        alphapref = alpha0;
        KG        = (cnstData.KE.* G_of_x(1:nap,1:nap)) / learningparams.lambda;
        h_of_x    = l_of_x;
        tol4y_E_y_I       = 0.001;
        maxit4y_E_y_I     = 2000;
        dopcg     = false;
        if dopcg  
           [alpha_new,flag,relres,iter] = pcg(KG,h_of_x,tol4y_E_y_I,maxit4y_E_y_I);
           assert(flag==0,'pcg didnot converge in computing y_E'); 
        else
        [alpha_new, histout, costdata]  = projbfgs(alpha_k,@f_lG_x_alpha,cnstData.up,cnstData.lo,tol4y_E_y_I,maxit4y_E_y_I);
        f_x       = costdata(end);
        end
        alpha_new(n_S + 1:end) =0;
    end
end
function [u_k] = projected_gradient_descent(u_0, first_order_oracle, project_func, Lip_const, max_iter, tol_grad)
    u_k     = u_0;
    u_pre   = u_k;
    h_k     = 1 / Lip_const;
    for k=1:max_iter 
       [g_u] = first_order_oracle(u_k); 
       normg(k) = norm(g_u); 
       
       if  normg(k) <= tol_grad
           return
       end
       u_k        = u_k - h_k* g_u;
       u_k        = project_func(u_k);
       dnorm      = norm(u_k-u_pre);
       if  dnorm <= tol_grad
           return
       end
       u_pre      = u_k;
    end
end
function [u_new] = accelerated_projected_gradient(u_0, first_order_oracle, project_func, Lip_const, mu_strcvx, max_iter, tol_grad)
    Lsqrt      = sqrt(Lip_const);
    musqrt     = sqrt(mu_strcvx);
    h_b        = (Lsqrt-musqrt)/(Lsqrt+musqrt);
    h_k        = 1/Lip_const;
    w_k        = u_0;
    u_new      = u_0;
    u_new_pre  = u_new;
    for k      = 1:max_iter 
        [g_u]     = first_order_oracle(w_k); 
        u_new     = w_k - h_k*g_u;       
        u_new     = project_func(u_new);
        du        = u_new-u_new_pre;
        normdu    = norm(du);
        if normdu <= tol_grad
            return
        end
        w_k       = u_new  + h_b*(du);
        u_new_pre = u_new;
    end
end