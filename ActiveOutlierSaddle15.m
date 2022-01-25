function [ALresult,model_set,Y_set]    = ActiveOutlierSaddle15(WMST_beforeStart, ModelAndData, learningparams, xTrain, yTrain)
%% global variables 
global cnstData
global WMST_variables
global WARMStart
ALresult.active  = true;
t1=tic;
[ALresult1, model_set1, Y_set1]= ActiveDConvexRelaxOutlier(WMST_beforeStart, ModelAndData, learningparams);
tAD= toc(t1)
x_opt     = model_set1.x_opt;
alpha_opt = ALresult1.alphav; 
tol_prox  = 1e-4;
verbose            = true;
%% Learning Parameters
%% Optimization Parameters
[optparams]            = optimaization_settings(learningparams);
%% Initialize Method
update_func            = {@no_acceleration_update, @nesterov_simple_update};
update_func_id         = 1;
progress_func          = @progress_report;
inexact_func           = {@inexact_mechanism1,@inexact_mechanism2,@inexact_mechanism_accelerated_k2}; % always method 1 works better. 
i_f                    = 1;
%% 8 and 5, working now
% 9 is based on the quadratic saddle algorithm and is very good for some
% problems, which I think the optimal point of unconstrained problem is in
% feasible region of the constraints. But unfortunately, this doesn't work
% for this problem. 
optparams.methodInd    = 8;%8;%5; % 1: Proximal Step Dual Averaging, 2: ProximalStepAlphaX, 3: ProximalStepXouterAlphaInner,4:Tseng_forwardbackwardforward,5:ProximalStepAlphaOuterXinner
methodfunc = {@ProximalStepDualAveraging, @ProximalStepAlphaX, @ProximalStepXOuterAlphaInner,...
              @Tseng_forwardbackwardforward, @nonsaddle_lssdp_ABCD_dual_noscale, @nonsaddle_nest_comp_dual,...
              @nest_comp_xouter_alphainner, @adjoint_lssdp_ABCD_dual_noscale, @accelerated_conic_saddle_algorithm};
objectivefunc.update_LHS_Mat_y_E_y_I = @ update_Qinv_LHS_Mat_y_E_y_I;
%% Initialize Optimization 
disp('Preprocessing...');
useWMST  = true;
[operators, y_EC, y_EV, y_IC, y_IV,x_st_IC, x_st_IV]                = getConstraints5(learningparams, useWMST, WMST_beforeStart, WMST_variables, ModelAndData.model);   

update_LHS_Mat_y_E_y_I = objectivefunc.update_LHS_Mat_y_E_y_I;
[operators.LHS_y_E, operators.LHS_y_I, operators.CHOL_H_y_E, operators.CHOL_H_y_I] ...
                = update_LHS_Mat_y_E_y_I(operators);
%% Checking correctness of getConstraint4
% [operators1]                = getConstraints3(learningparams); 
% toc
% df = norm(operators.A_EC-operators1.A_EC,1)+ norm(operators.A_EV-operators1.A_EV,1)...
%     +norm(operators.A_IC-operators1.A_IC,1) + norm(operators.A_IV-operators1.A_IV,1)...
%     +norm(operators.s_IC-operators1.s_IC,1) + norm(operators.s_IV-operators1.s_IV,1)...
%     +norm(operators.B_E-operators1.B_E,1)+ norm(operators.b_EV-operators1.b_EV,1)...
%     +norm(operators.b_EC-operators1.b_EC,1);
%%
disp('Starting Optimization ...');
ts = tic;
if ~WMST_beforeStart && WARMStart
   [x0, dualvars0, alpha0] = getWARMSTARTGlobal(y_EC, y_EV, y_IC, y_IV, x_st_IC, x_st_IV);
else
   [x0, dualvars0, alpha0] = initVariables();
end
x_pre        = x0;
dualvars_pre = dualvars0;
alpha_pre    = alpha0;
[optparams.L_x, optparams.L_alpha] = computeLipSchitz(learningparams); 
gamma_cond_x        = learningparams.rhox/optparams.L_x;
gamma_cond_alpha    = learningparams.rhoalpha/optparams.L_alpha;
inex_method         = 1;
[max_conv_meas, max_rel_gap, max_iter] = inexact_func{i_f}(inex_method, true, 1e-1, 1e-1, 100);
inexact_restart_duaration = 20;
sumof_sdpiter             = 0;
calphathreshold           = 0.1;
%% Starting Proximal Optimization loop
outerproxit = 1;
inexactit   = 1;
inexact_intact = 0;
converged      = false;
while outerproxit < optparams.maxouterprox &&~converged
    %% Proximal Point Iteration   
    [alpha_k,x_k, dualvars_k, solstatus ] ...
                = methodfunc{optparams.methodInd}(x0,alpha0, dualvars0, operators,optparams,learningparams, progress_func,verbose, max_conv_meas, max_rel_gap, max_iter);
    %% Update exactness parameters
    reconvalpha = norm(alpha_k-alpha0)/(1+norm(alpha0));
    reconvx     = sqrt(euclidean_dist_of_x(x_k,x0))/(1+sqrt(x_norm(x0)));
    d1          = sqrt(euclidean_dist_of_x(x_k, x_opt))/(1+sqrt(x_norm(x_opt)));                  
    dist_optimal_x(outerproxit) = d1;
    diffalpha   = alpha_k-alpha_opt;
    d2          = norm(diffalpha)/(1+norm(alpha_opt));
    dist_optimal_a(outerproxit) = d2;
    [max_conv_meas, max_rel_gap, max_iter] = inexact_func{i_f}(inex_method, false, max_conv_meas, max_rel_gap, max_iter,solstatus); 
    sumof_sdpiter               = sumof_sdpiter + solstatus.iter;
    optparams.stmax_iter = optparams.stmax_iter + 1;
    if  reconvalpha < calphathreshold % we need more accurate proximal steps
        %max_iter       = max_iter + 1;
        calphathreshold= 0.8*calphathreshold;
        optparams.stmax_iterADMM = optparams.stmax_iterADMM + 1;
    end
    report(outerproxit, verbose, ts);
    conv = max(reconvx, reconvalpha);
    if conv < tol_prox
        str = sprintf('converged in %13.7f seconds',toc(ts));
        disp(str);
        converged = true;
    end
    update_func{update_func_id}(outerproxit);
    %% next itertation            
    outerproxit = outerproxit + 1;
end
setWARMSTARTGlobal(x0, dualvars0, alpha0, operators);
[ALresult.queryind, model_set, Y_set] = get_query_label_of_x(x_k);
model_set.alpha_pv                    = alpha0;
model_set.constrain_instance_map      = operators.constrain_instance_map; 
%% End of Proximal loop
%% Show diagrams for performance measures
    function [max_conv_meas, max_rel_gap, max_iter] = inexact_mechanism1(inex_method, isstart, max_conv_meas, max_rel_gap, max_iter,solstatus)        
        if isstart 
            % do nothing just pass parameters to returned values
        elseif inex_method == 1 
            max_conv_meas               = max_conv_meas*inexactit/(inexactit+1);
            max_rel_gap                 = max_rel_gap  *inexactit/(inexactit+1);
            %do noting, just: max_iter                    = max_iter;
            inexactit = inexactit + 1;
        end
    end
    function [max_conv_meas, max_rel_gap, max_iter] = inexact_mechanism2(inex_method, isstart, max_conv_meas, max_rel_gap, max_iter,solstatus)        
        if isstart 
            % do nothing just pass parameters to returned values
        elseif inex_method == 1 
            if solstatus.iter == 1 % if just one inner iteration, it means inexact bound is higher than conv_meas of one iteration
                inexact_intact = inexact_intact + 1;
            else
                inexact_intact = 1;
            end
            if inexact_intact >= inexact_restart_duaration % if it is higher than conv_meas for a long time, restart inexact mechanism
                inexactit = floor(inexact_restart_duaration/2);%1;
                inexact_intact = 0;
            end
            max_conv_meas               = max_conv_meas*inexactit/(inexactit+1);
            max_rel_gap                 = max_rel_gap  *inexactit/(inexactit+1);
            %do noting, just: max_iter                    = max_iter;
            inexactit = inexactit + 1;
        end
    end
    function [max_conv_meas, max_rel_gap, max_iter] = inexact_mechanism_accelerated_k2(inex_method, isstart, max_conv_meas, max_rel_gap, max_iter,solstatus)        
        if isstart 
            % do nothing just pass parameters to returned values
        elseif inex_method == 1 
            delta  = 0.1;
            powe   = 2 + delta;
            max_conv_meas               = max_conv_meas*(inexactit)^powe/(inexactit+1)^powe;
            max_rel_gap                 = max_rel_gap  *(inexactit)^powe/(inexactit+1)^powe;
            %do noting, just: max_iter                    = max_iter;
            inexactit = inexactit + 1;
        end
    end
    function [x_G, dualvarsPre, alpha_alpha0] = initVariables()
        % if it is the first time this algorithm is started
        dualvarsPre.y_IV =  zeros(operators.n_AIV,1);
        dualvarsPre.y_EC =  zeros(operators.n_AEC,1);
        dualvarsPre.y_EV =  zeros(operators.n_AEV,1);
        dualvarsPre.y_IC =  zeros(operators.n_AIC,1);
        dualvarsPre.S    =  zeros(cnstData.nConic,1);
        dualvarsPre.Z    =  zeros(cnstData.nConic,1);
        dualvarsPre.v    =  zeros(operators.n_AIC + operators.n_AIV,1);

        n_I              =  operators.n_AIC + operators.n_AIV;
        alpha_0          = [rand(cnstData.n_S,1);zeros(cnstData.nappend,1)];
        %load('alphasave','alpha_0');
        w_obetapre       = zeros(cnstData.n_S,1);
        XSmall           = eye(cnstData.n_S);
        q                = ones(cnstData.nappend,1)/cnstData.n_q;
        Xq               = diag(q);
        XS               = [XSmall,zeros(cnstData.n_S,cnstData.nappend);...
                            zeros(cnstData.n_S,cnstData.nappend)',Xq];
        yqq              = [zeros(cnstData.n_S,1);q];
        X                = [XS,yqq;yqq',1];
        initL            = cnstData.initL(cnstData.initL>0);
        p(initL,1)         = ones(cnstData.n_l,1)*cnstData.lnoiseper/100;
        p(cnstData.unlabeled,1) = ones(cnstData.n_u,1)*cnstData.onoiseper/100;
%         p                = ones(cnstData.n_S,1)/cnstData.n_o;  
        a                = zeros(cnstData.n_S,1);
        g                = zeros(cnstData.n_S,1);
        u                = [reshape(X,cnstData.nSDP*cnstData.nSDP,1);p];
        vpre             = zeros(n_I,1);
        stpre            = vpre;
        % Spre     = u;% this is zero for now
        upre             = u;% this is zero for now
        x_G.u            = upre;
        x_G.st           = stpre;
        x_G.w_obeta      = w_obetapre;
        alpha_alpha0     = alpha_0;
    end
    function report(iter,verbose, time_handle)
       moditer = 5;
       if (mod(iter,moditer)==1)&&verbose
             strtitle = sprintf('iter| x conv  |alphaconv| inner:#iter |inner_Conv|inner_Gap|      f_k|    f_opt|  Time');
             disp(strtitle);
       end
       timespent = toc(time_handle);
       if verbose
            str = sprintf('%4.0d|%9.6f|%9.6f|         %4.0d|%10.7f|%9.6f|%9.6f|%9.6f|%13.7f',...
                           iter, reconvx, reconvalpha,solstatus.iter,solstatus.conv_meas,solstatus.rel_gap,d1,d2,timespent);
            disp(str);
       end 
    end
    function no_acceleration_update(k)
        x_pre   = x_k;
        dualvars_pre = dualvars_k;
        alpha_pre    = alpha_k;
        %% Update Proximal Point and Previous Lagrange Values
        dualvars0  = dualvars_k;
        x0         = x_k;
        alpha0     = alpha_k;
    end
    function nesterov_simple_update(k)
        y_dualvars_k.y_EC = dualvars_k.y_EC      + (k-1)/(k+1)*(dualvars_k.y_EC-dualvars_pre.y_EC);
        y_dualvars_k.y_EV = dualvars_k.y_EV      + (k-1)/(k+1)*(dualvars_k.y_EV-dualvars_pre.y_EV);
        y_dualvars_k.y_IC = dualvars_k.y_IC      + (k-1)/(k+1)*(dualvars_k.y_IC-dualvars_pre.y_IC);
        y_dualvars_k.y_IV = dualvars_k.y_IV      + (k-1)/(k+1)*(dualvars_k.y_IV-dualvars_pre.y_IV);
        y_dualvars_k.S    = dualvars_k.S         + (k-1)/(k+1)*(dualvars_k.S   -dualvars_pre.S);
        y_dualvars_k.Z    = dualvars_k.Z         + (k-1)/(k+1)*(dualvars_k.Z   -dualvars_pre.Z);
        y_dualvars_k.v    = dualvars_k.v         + (k-1)/(k+1)*(dualvars_k.v   -dualvars_pre.v);
        y_alpha_k         = alpha_k              + (k-1)/(k+1)*(alpha_k        -alpha_pre);
        y_x_k.u           = x_k.u                + (k-1)/(k+1)*(x_k.u          -x_pre.u);
        y_x_k.st          = x_k.st               + (k-1)/(k+1)*(x_k.st         -x_pre.st);
        y_x_k.w_obeta     = x_k.w_obeta          + (k-1)/(k+1)*(x_k.w_obeta    -x_pre.w_obeta);
        
        x_pre   = x_k;
        dualvars_pre = dualvars_k;
        alpha_pre    = alpha_k;
        %% Update Proximal Point and Previous Lagrange Values
        dualvars0  = y_dualvars_k;
        x0         = y_x_k;
        alpha0     = y_alpha_k;
    end
end