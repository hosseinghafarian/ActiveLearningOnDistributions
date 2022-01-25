function [alpha_k, x_k, dualvars_k, solstatus ] = accelerated_conic_saddle_algorithm(x_G, alpha_0, dualvars_0, operators,optparams,learningparams, progress_func,verbose, max_conv_meas, max_rel_gap, max_iter)
% Global variables 
global cnstData
% load('xalpha','x_next','alpha_next');
% x_G     = x_next;
% alpha_0 = alpha_next; 
%% Setting objective function components 
objectivefunc.primal      = @primal_objective;
objectivefunc.dual        = @  dual_objective_func;
objectivefunc.dualsplit   = @  dual_objective_split;
objectivefunc.LHSRHS      = @  dual_objective_split_LHS_RHS;
objectivefunc.update_LHS_Mat_y_E_y_I = @ update_Qinv_LHS_Mat_y_E_y_I;
objectivefunc.dist_x_opt  = @relative_dist_to_x_opt;
objectivefunc.ProofChecker= @NestCompProofChecker; 
BCD_dual_solver = {@solve_BCDDetail_dualsub_problems, @solve_BCD2by3_solver_dualsub_problems, @solve_BCDDetail_Zvblocksolver};
options.usexHessian = false;
options.useuHessian = false;
options.usexpcg     = false;
options.useupcg     = false;
options.pcgtolstart = 10^-1;
options.pcgtolend   = 10^-6;
options.useuL_xu    = true;
options.usexL_xu    = true;
%% setting initial values for improvement measures
Dobjective                = zeros(max_iter,1);
Pobjective                = zeros(max_iter,1);
etaIC                     = zeros(max_iter,1);
etaCone                   = zeros(max_iter,1);
etaEC                     = zeros(max_iter,1);
etagap                    = zeros(max_iter,1);
etalpha                   = zeros(max_iter,1);
etaall                    = zeros(max_iter,1);
L_xx                      = learningparams.rhox;
L_alphaalpha              = learningparams.rhoalpha+norm(cnstData.KE);
L_alphax                  = 1; % this value is very much different for values of x and alpha, so, it must be computed locally. ???? 
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
yEyItol         = optparams.tol_ADMM;
yEyImax_iter    = 100; %pcg maxiteration
yEyIsoltype     = 3;   % compute y_E and y_I using 1: optimization using yalmip , 2: solve Ax=b using pcg ,3:solve Ax=b using cholesky factorization. 

sub_solvetype      = 1;
%% initilizing loop
% fetch starting values for x, dualvars and alpha
[x_0 ]        = update_x_0_alpha(alpha_0, x_G);
alpha_ktil    = alpha_0;
n_alpha       = numel(alpha_0);
[y_ECtil,y_EVtil,y_ICtil,y_IVtil,Stil,Ztil,vtil] = load_dual_vars(dualvars_0);
n_EC          = numel(y_ECtil);
n_EV          = numel(y_EVtil);
n_IC          = numel(y_ICtil);
n_IV          = numel(y_IVtil);
n_Stil        = numel(Stil);
n_Ztil        = n_Stil;
n_vtil        = numel(vtil);
n_dual        = n_EC + n_EV + n_IC + n_IV + n_Stil + n_Ztil + n_vtil;
n_x_u         = numel(x_0.u);
n_x_wo        = numel(x_0.w_obeta);
n_x_st        = numel(x_0.st);
n_primal      = n_x_u + n_x_wo + n_x_st;
grad_x_app    = zeros(n_primal + n_dual, 1);
Hessian_alphax_app = zeros(n_alpha, n_primal+n_dual);
x_k_app       = [x_0.u;x_0.w_obeta;x_0.st;y_ECtil;y_EVtil;y_ICtil;y_IVtil;Stil;Ztil;vtil];
ind_start     = 0;
ind_x_u       = ind_start+1:ind_start+n_x_u;         ind_start = n_x_u;
ind_x_wo      = ind_start+1:ind_start+n_x_wo;        ind_start = ind_start + n_x_wo;
ind_x_st      = ind_start+1:ind_start+n_x_st;        ind_start = ind_start + n_x_st;
ind_y_EC      = ind_start+1:ind_start+n_EC;          ind_start = ind_start + n_EC;
ind_y_EV      = ind_start+1:ind_start+n_EV;          ind_start = ind_start + n_EV;
ind_y_IC      = ind_start+1:ind_start+n_IC;          ind_start = ind_start + n_IC;
ind_y_IV      = ind_start+1:ind_start+n_IV;          ind_start = ind_start + n_IV;
ind_Stil      = ind_start+1:ind_start+n_Stil;        ind_start = ind_start + n_Stil;
ind_Ztil      = ind_start+1:ind_start+n_Stil;        ind_start = ind_start + n_Ztil;
ind_vtil      = ind_start+1:ind_start+n_vtil;        ind_start = ind_start + n_vtil;
%%Attention size of x_0.st?

% set starting  and previous values for variables 
alpha_k       = alpha_ktil;
alpha_pre     = alpha_ktil;
x_k_app_pre   = x_k_app;
    
    function [x_0 ] = update_x_0_grad(a_x, x_hat_app, L_A_x)
       g_acc_x.u       = a_x(ind_x_u);
       g_acc_x.w_obeta = a_x(ind_x_wo);
       g_acc_x.st      = a_x(ind_x_st);
       beta_kx         = L_A_x;
       x_0.u           = (1/beta_kx)*(- g_acc_x.u                     + beta_kx*x_hat_app(ind_x_u ));
       x_0.w_obeta     = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_hat_app(ind_x_wo));
       x_0.st          = (1/beta_kx)*(- g_acc_x.st                    + beta_kx*x_hat_app(ind_x_st)); 
    end
    function [ y_EC, y_EV, y_IC, y_IV, S, Z , v] = extract(x_hat_app)
       y_EC = x_hat_app(ind_y_EC);
       y_EV = x_hat_app(ind_y_EV);
       y_IC = x_hat_app(ind_y_IC);
       y_IV = x_hat_app(ind_y_IV);
       S    = x_hat_app(ind_Stil);
       Z    = x_hat_app(ind_Ztil);
       v    = x_hat_app(ind_vtil);
    end
    function [grad_x_app_r, grad_alpha, Hessian_alphax_app_r ] = semi_second_order(x_hat_app, alpha_hat)
       %[f,g_x,grad_alpha]  = f_xAlpha_grad(x_G,alpha_k,learningparams);
       x_hat.u                 = x_hat_app(ind_x_u);
       x_hat.w_obeta           = x_hat_app(ind_x_wo);
       x_hat.st                = x_hat_app(ind_x_st);
       [g_x, grad_alpha, H_alphax_u, H_alphax_w_obeta, H_alphax_st]  = f_xAlpha_regularized_semiHessian(x_hat,alpha_hat,learningparams, x_G, alpha_0);
       grad_x_app(1:n_primal)  = [g_x.u;g_x.w_obeta;g_x.st];
       Hessian_alphax_app(:,1:n_primal)  = [H_alphax_u, H_alphax_w_obeta, H_alphax_st];
       grad_x_app_r            = grad_x_app;
       Hessian_alphax_app_r    = Hessian_alphax_app';
    end
    function [x_k_app, alpha_k] = prox_operator(L_xx, L_alphaalpha, L_xalpha, L_A_x, L_A_alpha, a_x, A_x, a_alpha, A_alpha, x_hat_app, alpha_hat, usexL_xalpha, usealphaL_xalpha)
       % extract x_hat_app parts
       [ y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Ztil , vtil]  = extract(x_hat_app);  
       [x_0 ]           = update_x_0_grad(a_x, x_hat_app, L_A_x);
       % proximal step on x
       [ y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I] = BCD_dual_solver{sub_solvetype}();   
       [x_k ]           = x_conv_from_dual_fullproject(y_EC,y_EV,y_IC,y_IV,S,Z,v, x_0, operators);
       % proximal step on alpha
       [alpha_k    ]    = proxf_alpha(L_A_alpha, a_alpha, optparams, alpha_hat);
       %[f,g_x,g_alpha]  = f_xAlpha_grad(x_k,alpha_ktil,learningparams);
       %Dobjective(iter) = dualsplitfunc(y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
       x_k_app          = [x_k.u;x_k.w_obeta;x_k.st;y_EC;y_EV;y_IC;y_IV;S;Z;v];
    end
    function [eta] = compConvMeasures(meastype, x_k_app, alpha_k, x_hat_app, alpha_hat,iter)
        [ y_EC, y_EV, y_IC, y_IV, S, Z , v]  = extract(x_k_app);
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];
        w_obetav     = x_0.w_obeta + cnstData.Hinv*(operators.B_EV'*y_EV+operators.B_IV'*y_IV);
        X            = proj_oncones(x_0.u+Ay+Z,cnstData.nSDP,cnstData.n_S,0);   
        Xp           = proj_oncones(x_0.u+Ay+S+Z,cnstData.nSDP,cnstData.n_S,0);   
        Y            = proj_onP(x_0.u+Ay+S,cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
        st           = min(x_0.st-[y_IC;y_IV],s_I);
        Pobjective(iter)= primal_func(x_k, x_0, operators, learningparams, optparams);
        etaEC(iter)     = norm([operators.A_EC*X-operators.b_EC;operators.A_EV*X+operators.B_EV*w_obetav-operators.b_EV])/(1+norm([operators.b_EC;operators.b_EV]));
        etaIC(iter)     = norm(st+[-operators.A_IC*X;-operators.A_IV*X-operators.B_IV*w_obetav])/(1+norm(st));
        etaCone(iter)   = norm(X-Y)/(1+norm(X));
        %etagap is not computed based on my own formulation of the problem. 
        etagap(iter)    = abs(Dobjective(iter)-Pobjective(iter))/(1+abs(Pobjective(iter))+abs(Dobjective(iter)));
        etalpha(iter)   = norm(alpha_k-alpha_pre)/(1+norm(alpha_k));
        etaall(iter)    = max(max ( max(etaEC(iter),etaCone(iter)),etaIC(iter)),etalpha(iter));
        eta             = etaall(iter);
    end
    function [fvalue] = fxalpha(x_hat_app, alpha_hat)
        x_hat.u      = x_hat_app(ind_x_u);
        x_hat.w_obeta= x_hat_app(ind_x_wo);
        x_hat.st     = x_hat_app(ind_x_st);
        fvalue       = f_of_xAndAlpha_regularized(x_hat,alpha_hat,learningparams,3, x_G, alpha_0);
    end
    function_arguments.semi_second_order = @semi_second_order;
%     function_arguments.Hessian_x         = @second_order_x;
%     function_arguments.Hessian_u         = @second_order_u;
    function_arguments.prox_operator     = @prox_operator;
    function_arguments.objective_func    = @fxalpha;
    function_arguments.converge_measure  = @compConvMeasures;
    x_0_app =  [x_0.u;x_0.w_obeta;x_0.st;y_ECtil;y_EVtil;y_ICtil;y_IVtil;Stil;Ztil;vtil];
    max_iter      = 100;
    max_conv_meas = 10^-4;
    max_rel_gap   = 10^-4;
    [y,v] = saddle_solver_ver2(options, x_0_app, alpha_0, function_arguments, n_primal, n_alpha, L_xx, L_alphaalpha, L_alphax, max_conv_meas, max_rel_gap, max_iter);
    diff  = norm(y-x_star)+norm(v-u_star);

%% setup return values for alpha_k, x_k and dualvars_k
    dualvars_k                 = dualvar_conv(y_EC, y_EV, y_IC, y_IV, S, Z ,  v );
    x_k                        = x_conv_from_dual_fullproject(y_EC, y_EV, y_IC, y_IV, S, Z, v, x_0, operators);
    function [y_EC, y_EV, y_IC, y_IV, S, Z , v, normg_y_E, normg_y_I ] = solve_BCDDetail_dualsub_problems
        star         = 1;
        Ay           = A'*[y_ECtil;y_EVtil;y_ICtil;y_IVtil];
        R            = Ay + Stil + x_0.u;
        [Z,v]        = projon_Conestar(cnstData.extendInd,R, x_0.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter, ...
                                       y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter,...
                                       y_EC, y_EV, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];                          
        S            = -(Ay+Z+x_0.u);
        S            = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_E    = norm(g_y_E);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter,...
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
        [y_EC, y_EV, y_IC, y_IV, S] = solve_BCDDetail_SyIyEblock_problems(Z,v);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_I    = norm(g_y_I); normg_y_E    = norm(g_y_E);                         
    end
    function [y_EC, y_EV, y_IC, y_IV, S, normg_y_E, normg_y_I ]        = solve_BCDDetail_SyIyEblock_problems(Z, v)
        star         = 1;
        Ay           = A'*[y_ECtil;y_EVtil;y_ICtil;y_IVtil];
        R            = Ay + Stil + x_0.u;
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter, ...
                                       y_ECtil, y_EVtil, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter,...
                                       y_EC, y_EV, y_ICtil, y_IVtil, Stil, Z , v,...
                                       x_0, operators, learningparams,optparams);
        Ay           = A'*[y_EC;y_EV;y_IC;y_IV];                          
        S            = -(Ay+Z+x_0.u);
        S            = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
        [y_IC, y_IV] = proxLmu_y_I(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter,...
                                       y_EC, y_EV, y_IC, y_IV, S, Z , v,...
                                       x_0, operators, learningparams,optparams);
        [f_dual, g_y_E, g_y_I, g_S, g_Z, g_v] = dual_regwo_objective_split_grad(b_E,y_EC, y_EV, y_IC, y_IV, S, Z , v, x_0, operators, learningparams, optparams);
        normg_y_E    = norm(g_y_E);
        [y_EC, y_EV] = proxLmu_y_E(objectivefunc,yEyIsoltype,yEyItol,yEyImax_iter,...
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
        rhop      = 10;%learningparams.rhoalpha;
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
            diag_lKGwithoutx= -l_star + KGwithoutx* alpha_new;
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
    function [alpha_new]  = proxf_alpha(L_alphaalpha, a_alpha, optparams, alpha_hat)
        %% it is very interesting that a few iterations of accelerated_first
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
        %%
        second_order_alpha      = false;
        accelerated_first_alpha = false;
        if second_order_alpha 
            [alpha_new]  = sec_order_proxf_alpha(learningparams, optparams, alpha_hat, x_k);
        else
            if accelerated_first_alpha
                optimizer_func = @accelerated_projected_gradient;
            else
                optimizer_func = @projected_gradient_descent;
            end 
            [alpha_new]  = first_order_proxf_alpha(optimizer_func,  L_alphaalpha, a_alpha, optparams, alpha_hat, x_k); 
        end
    end
    function [alpha_new]  = first_order_proxf_alpha(optimizer_func, L_alphaalpha, a_alpha, optparams, alpha_hat, x_k)
        %Attention: when rhop is a not a value greater than the
        %largest eigen values of the matrix KG, alpha_k goes rogue! In fact
        %it oscilates between two values. Why?
        graditer  = 50;
        tol_grad   = 10^-5;
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        function [gout ] = grad_alpha(alpha_new)
            gout      = a_alpha + L_alphaalpha*(alpha_new-alpha_hat);
        end
        function [alpha_prj ] = project_alpha(alpha_new)
            alpha_new = max(alpha_new, cnstData.lo);
            alpha_prj = min(alpha_new, cnstData.up);
        end
        L_a_local    = L_alphaalpha;
        [alpha_new ] = optimizer_func(alpha_hat, @grad_alpha, @project_alpha, L_a_local, graditer, tol_grad);
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
        yEyItol       = 0.001;
        yEyImax_iter     = 2000;
        dopcg     = false;
        if dopcg  
           [alpha_new,flag,relres,iter] = pcg(KG,h_of_x,yEyItol,yEyImax_iter);
           assert(flag==0,'pcg didnot converge in computing y_E'); 
        else
        [alpha_new, histout, costdata]  = projbfgs(alpha_k,@f_lG_x_alpha,cnstData.up,cnstData.lo,yEyItol,yEyImax_iter);
        f_x       = costdata(end);
        end
%         alpha_new(n_S + 1:end) =0;
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