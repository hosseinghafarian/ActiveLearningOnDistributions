function [alpha_hatbar, x_hatbar, dualvars, solstatus ] = ProximalStepDualAveraging(x_G,alpha_0,dualvars0, operators, optparams,learningparams, progress_func,verbose)
    global cnstData
    
    global optimals
    %alpha_opt = optimals.alpha_opt ;

    moditer         = 2;
    max_conv_meas   = 10^-4;
    max_rel_gap     = 10^-4;
    max_iter        = 200;
    %% Setting starting values of variables
    %%%%%%%%%%%%%%%%%%%%%For now assume that we have alpha_opt, let see what happens.
    x_0             = x_G;
    x_k             = x_0;
    x_hat           = x_0;
    x_hatbar        = x_hat;
    dualvars_k      = dualvars0;
    dualvars_hat    = dualvars0;
    dualvars_hatbar = dualvars_hat;
    
    objectivefunc.primal      = @primal_objective;
    objectivefunc.dual        = @  dual_objective_func;
    objectivefunc.dualsplit   = @  dual_objective_split;
    objectivefunc.LHSRHS      = @  dual_objective_split_LHS_RHS;
    objectivefunc.update_LHS_Mat_y_E_y_I = @ update_Qinv_LHS_Mat_y_E_y_I;
    objectivefunc.dist_x_opt  = @relative_dist_to_x_opt;
    objectivefunc.ProofChecker = @NestCompProofChecker; 
    alpha_k         = alpha_0;
    alpha_hat       = alpha_0;
    alpha_hatbar    = alpha_hat;
    %% Retriving learningparameters for using in iterations
    BCGD            = true;
    [L_x, L_alpha, L_w_obeta, D_x, D_alpha, p_beta_Alpha,L,D,gamma_b]     = lipschitz_of_xAndAlpha(learningparams);
    p_beta_Alpha    = 0.27;
    gamma_b         = 0.05;
    %% Starting values for Nesterov's coeff's
    %gamma_b         = 1.2;
    S_k             = 1;  %start from 0,\lambda_0=1,S_0=\lambda_0;
    A_pre           = 0;
    g_acc_x.u       = zeros(cnstData.nConic,1);
    g_acc_x.w_obeta = zeros(cnstData.n_S,1);
    g_acc_x.st      = 0;
    g_acc_alpha     = zeros(cnstData.nSDP-1,1);
    accumf          = 0;
    accumfgrad      = 0;
    accumgrada      = 0;
    %% Exactness and max iteration parameters

    convergemeasure = zeros(optparams.stmax_iter,4);
    dist            = zeros(optparams.stmax_iter,4);
    f_val           = zeros(optparams.stmax_iter,2);
    %% Starting loop    
    converged       = false;
    i               = 1; 
    while ~converged && (i<= 500)% optparams.stmax_iter)
        t      = cputime;
        %% concentrate on gradient of function for x and alpha, that is g_k=(g_x,-g_\alpha)
        %[c_u, c_beta, c_s, c_alpha]= xalpha_grad(x_k, alpha_k,x_0,alpha_0, learningparams);
        [f,g_x,g_alpha]   = f_xAlpha_grad(x_k,alpha_k,learningparams);
        g_alpha           = -g_alpha;
        %% Choose lambda_k: parameter for s_{k+1} = s_{k} + \lambda_k* g_k
        % simple averages
        lambda_k          =   1;
        % weighted averages
        % lambda_k          = 1/norm([c_u;c_beta;c_s;c_alpha]);
        A_curr            =   A_pre + lambda_k;
        %% Compute Weighted sum of g: s_{k+1} = s_{k} + \lambda_k* g_k
        g_acc_x.u       = g_acc_x.u       + lambda_k* g_x.u;%c_u;
        g_acc_x.w_obeta = g_acc_x.w_obeta + lambda_k* g_x.w_obeta;
        g_acc_x.st      = g_acc_x.st      + lambda_k* g_x.st;
        g_acc_alpha     = g_acc_alpha     + lambda_k* g_alpha;
        %% Choose \beta_{k+1} such that \beta_{k+1} >= \beta_k
        if i==1
            betahat_k = 1;
        else
            betahat_k = betahat_k+1/betahat_k;
        end;
        beta_k        = gamma_b* betahat_k;
        %% Computing v_k : wmstalpha,alph0,accumGrad,accumf,accumgrada,rho,A_curr,lo,up,tol,maxit
        [x_next, dualvars_next, alpha_next] = argminpsi_x(BCGD, x_0, alpha_0, x_k, alpha_k, g_acc_x, g_acc_alpha, beta_k, learningparams, optparams);           
        %% Compute: S_k = \sum_{i=0}^k \lambda_i,              
        %           \hat{x}_{k+1} = \frac{1}{S_k} \sum_{i=0}^k \lambda_i x_i, 
        %           s_{k+1} = \sum_{i=0}^k \lambda_i g_i,-> accumGrad_...
        %           \hat{s}_{k+1} = \frac{1}{S_k} s_{k+1}
        S_k              = S_k           + lambda_k;
        x_hat.u          = x_hat.u       + lambda_k*x_next.u;
        x_hat.w_obeta    = x_hat.w_obeta + lambda_k*x_next.w_obeta;
        x_hat.st         = x_hat.st      + lambda_k*x_next.st;
        dualvars_hat.y_EC= dualvars_hat.y_EC + lambda_k*dualvars_next.y_EC;
        dualvars_hat.y_EV= dualvars_hat.y_EV + lambda_k*dualvars_next.y_EV;
        dualvars_hat.y_IC= dualvars_hat.y_IC + lambda_k*dualvars_next.y_IC;
        dualvars_hat.y_IV= dualvars_hat.y_IV + lambda_k*dualvars_next.y_IV;
        dualvars_hat.S   = dualvars_hat.S    + lambda_k*dualvars_next.S;
        dualvars_hat.Z   = dualvars_hat.Z    + lambda_k*dualvars_next.Z;
        dualvars_hat.v   = dualvars_hat.v    + lambda_k*dualvars_next.v;
        alpha_hat        = alpha_hat     + lambda_k*alpha_next;
        
        x_hatbar_pre     = x_hatbar;
        dualvars_hatbarpre = dualvars_hatbar;
        
        alpha_hatbar_pre = alpha_hatbar;
        x_hatbar.u       = x_hat.u/S_k;
        x_hatbar.w_obeta = x_hat.w_obeta/S_k;
        x_hatbar.st      = x_hat.st/S_k;
        dualvars_hatbar.y_EC = dualvars_hat.y_EC /S_k;
        dualvars_hatbar.y_EV = dualvars_hat.y_EV /S_k;
        dualvars_hatbar.y_IC = dualvars_hat.y_IC /S_k;
        dualvars_hatbar.y_IV = dualvars_hat.y_IV /S_k;
        dualvars_hatbar.S    = dualvars_hat.S    /S_k;
        dualvars_hatbar.Z    = dualvars_hat.Z    /S_k;
        dualvars_hatbar.v    = dualvars_hat.v    /S_k;
        
        alpha_hatbar     = alpha_hat/S_k;
        
        [Xapprox,p,q,qyu,w_obeta,st]  = getParts(x_hatbar);
        %% updating performance measures
        % warning: How to update dualvars, ergodically?
        [convergemeasure(i,:), dist(i,:), f_val(i,:)] ...
            = progress_func(learningparams,i,moditer, verbose, x_hatbar, x_hatbar_pre, dualvars_next, dualvars_k, alpha_hatbar, alpha_hatbar_pre, x_k, alpha_k); 
        %updateandprintconvmeasure(verbose);
        
%         if i>1 && convergemeasure(i,4) < 0.1*convergemeasure(i-1,4)
%             converged = true;
%             %optparams.stmax_iterADMM = optparams.stmax_iterADMM +1;
%         end
        %% Prepare for the next iterate
        x_k                  = x_next;
        dualvars_k           = dualvars_next;
        alpha_k              = alpha_next;
        A_pre                = A_curr;
        i                    = i + 1;
        if mod(i,5)==0, max_iter = max_iter + 1; end
    end  
    i = i+1;
    %% Computing Error for the last inner teration using method of paper, Kolossoski ->goto ActiveOutlierSaddle8 and previous
    function [x_next, dualvars_next, alpha_next]     = argminpsi_x(BCGD, x_0, alpha_0,x_k,alpha_k, g_acc_x, g_acc_alpha, beta_k, learningparams,optparams)
            % set a parameter p for  d(x)=p d(y) + (1-p)*d(alpha), d(y)=
            % norm(y-y_0)^2;d(alpha)=norm(alpha-alpha0)^2;
            % compute x_{k+1} = argmin_{x\in Q} \{-<(-s),x>+\beta_k* d(x)\}
            % this will become two separate optimization problems
            % regarding, y and alpha. 
            p             = p_beta_Alpha;
            beta_kalpha   = (1-p)*beta_k;
            beta_kx       = p*beta_k;
            [alpha_next ] = argminPi_alpha(alpha_0, alpha_k, g_acc_alpha, beta_kalpha);
            alpha_next(cnstData.extendInd) = 0;
            [x_next, dualvars_next] = argminPi_x(BCGD, x_0, x_k, g_acc_x, beta_kx,learningparams,optparams);
    end
    function [alpha_next ]              = argminPi_alpha(alpha_0, wmstalpha, accumGrad_alpha, beta_kalpha)
            global    proxParam;
            global    alphProx;
            global    accumGradProx;
            % objective function:  accumGradProx^T*\alpha + \frac{proxParam}{2}*norm(alphav-alphProx)^2
            % This is in function:psi_alpha
            tol             = 10e-4;
            maxit           = 100;
            accumGradProx   = accumGrad_alpha;
            proxParam       = beta_kalpha/2;
            alphProx        = alpha_0;
            [alpha_next,histout,costdata,iterAlpha] = projbfgs(wmstalpha,@psi_alpha,cnstData.up,cnstData.lo,tol,maxit); % This is two order of magnitude faster than projected gradient
            %[x,histout,costdata,iterAlpha] = gradproj(wmstalpha,@psi_alpha,up,lo,tol,maxit);
    end
    function [x_next,dualvars_next]        = argminPi_x(BCGD, x_0,  x_next, g_acc_x, beta_kx,learningparams,optparams)
            if ~BCGD 
                %[x_next,dualvars_k, comp, feas, eqIneq] = solvePsi_k4_copy(objectivefunc,x_0, g_acc_x, beta_kx,operators,learningparams,optparams);
                [x_next,dualvars_next, comp, feas, eqIneq] = solvePsi_k5     (objectivefunc,x_0, g_acc_x, beta_kx,operators,learningparams,optparams);        
            else
                %[x_k1,dualvars_k1, comp, feas, eqIneq] = solvePsi_k5     (objectivefunc,x_0, g_acc_x, beta_kx,operators,learningparams,optparams);        
                [ dualvars_next, x_next, solstatus ] = lssdpplus_innerx_ABCD_dual_noscale(max_conv_meas, max_rel_gap, max_iter, objectivefunc, ...
                                                                                         x_0, dualvars_hatbar, g_acc_x, beta_kx, operators,learningparams,optparams);
                %dx = euclidean_dist_of_x(x_k, x_k1)/x_norm(x_k1);
%                 dd = euclidean_dist_of_duals(dualvars_k, dualvars_k1);
                %[x_k,dualvars_k] = solveBCGDDualProblem(x_0, g_acc_x, beta_kx,operators,learningparams,optparams);
            end
    end
    function [x_next,dualvars]        = T_L(BCGD,x_curr, dualvars_curr, c_u,c_beta,c_s, learningparams, optparams, x_G)
            accumfgrad  = 0;
            arho        = learningparams.rhox;
            Lx          = norm(cnstData.K);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For now, it must be calculated exactly. 
            mu_reg      = Lx*5;
            if ~BCGD 
                [x_next,dualvars]=solvePsi_k3(x_curr,operators,dualvars_curr,...
                            x_G,c_u,c_beta,c_s,accumfgrad,learningparams,optparams,arho,mu_reg);
            else
                [x_next,dualvars]  = solveBCGDDualProblem(x_curr,dualvars_curr, x_G,c_u,c_beta,c_s,operators,learningparams,optparams,arho,mu_reg);
            end
    end
    function [v_k_x,dualvars]         = solveBCGDDualProblem(x_0, g_acc_x, beta_kx,operators,learningparams,optparams)
            
            Ghat.u    = (1/beta_kx)   *(- g_acc_x.u                     + beta_kx*x_0.u      );
            Ghat.w_obeta = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_0.w_obeta);
            Ghat.st   = (1/beta_kx)   *(- g_acc_x.st                    + beta_kx*x_0.st     );
            
            gscale         = 1;
            %[gscale,operators] = scaleProblem(learningparams,unscoperators,Ghat);
            
            s_I            = [operators.s_IC;operators.s_IV];
            star           = 1; 
            y_ECtil        = dualvars.y_EC;
            y_EVtil        = dualvars.y_EV;
            y_ICtil        = dualvars.y_IC;
            y_IVtil        = dualvars.y_IV;
            Stil           = dualvars.S   ;

            soltype        = 1;
            %% Step 1
            Ay             = operators.A_EC'* y_ECtil + operators.A_IC'* y_ICtil + operators.A_EV'* y_EVtil + operators.A_IV'* y_IVtil;
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            R              = Ay + Stil + Ghat.u;
            [Xapprox,p,q,qyu]     = getu_Parts(R); 
            [Z,v]          = projon_Conestar(cnstData.extendInd,R, x_G.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
            [Xapprox,p,q,qyu]     = getu_Parts(Z);  
            [y_EC,y_EV]    = x_proxLmu_y_E(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, Stil, y_ECtil, y_EVtil, y_ICtil, y_IVtil,...
                                           Ghat, operators);

            [y_IC,y_IV]    = x_proxLmu_y_I(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, Stil, y_EC, y_EV, y_ICtil, y_IVtil,...
                                           Ghat, operators);

            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;                     
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            S              = -(Ay+Z+Ghat.u);
            S              = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
            [Xapprox,p,q,qyu]     = getu_Parts(S); 
 
            [y_IC,y_IV]    = x_proxLmu_y_I(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                           Ghat, operators);
            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            [y_EC,y_EV]    = x_proxLmu_y_E(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                           Ghat, operators);
            %% 
            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;  
            [Xapprox,p,q,qyu]     = getu_Parts(Ay);
            Xp             = proj_oncones(Ghat.u+(Ay+S+Z),cnstData.nSDP,cnstData.n_S,0);   
            [Xapprox,p,q,qyu]     = getu_Parts(Xp); 
            Y              = proj_onP(Ghat.u+(Ay+S),cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
            [Xapprox,p,q,qyu]     = getu_Parts(Y); 
            v_k_x.st       = gscale*min(Ghat.st-[y_IC;y_IV],s_I);
            v_k_x.w_obeta  = gscale*(Ghat.w_obeta + cnstData.Qinv*(operators.B_EV'*y_EV+operators.B_IV'*y_IV));
            %v_k_x.u        = gscale*proj_oncones(Ghat.u+(Ay+Z),cnstData.nSDP,cnstData.n_S,0);   
            v_k_x.u        = gscale*proj_oncones(Ghat.u+(Ay+S+Z),cnstData.nSDP,cnstData.n_S,0); 
            
            dualvars.y_EC  = y_EC;
            dualvars.y_EV  = y_EV;
            dualvars.y_IC  = y_IC;
            dualvars.y_IV  = y_IV;
            dualvars.S     = S;        
    end    
    function [alpha_new, f_x]         = proxf_Alpha(learningparams, optparams, alpha0, x_k)
        global KG;
        global h_of_x;
        global alphapref;
        global rhop;
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        [l_of_x, G_of_x] = x_obj_get_lG_of_x(x_k);
        rhop      = learningparams.rhoalpha;
        alphapref = alpha0;
        KG        = (cnstData.KE.* G_of_x(1:nap,1:nap)) / learningparams.lambda;
        h_of_x    = [l_of_x;zeros(nap-n_S,1)];
        tol       = 0.001;
        maxit     = 2000;
        [alpha_new, histout, costdata]  = projbfgs(alpha0,@f_lG_x_alpha,cnstData.up,cnstData.lo,tol,maxit);
        f_x       = costdata(end);
    end
    function update_v_k_x_dual()
        x_curr.u           = A_pre/A_curr * x_k.u              + a_k / A_curr* x_next.u;
        x_curr.st          = A_pre/A_curr * x_k.st             + a_k / A_curr* x_next.st;
        x_curr.w_obeta     = A_pre/A_curr * x_k.w_obeta        + a_k / A_curr* x_next.w_obeta;    
        dualvars_curr.y_EC = A_pre/A_curr * dualvars_k.y_EC    + a_k / A_curr* alpha_next.y_EC;
        dualvars_curr.y_EV = A_pre/A_curr * dualvars_k.y_EV    + a_k / A_curr* alpha_next.y_EV;
        dualvars_curr.y_IV = A_pre/A_curr * dualvars_k.y_IV    + a_k / A_curr* alpha_next.y_IV;
        dualvars_curr.y_IC = A_pre/A_curr * dualvars_k.y_IC    + a_k / A_curr* alpha_next.y_IC;
        dualvars_curr.S    = A_pre/A_curr * dualvars_k.S       + a_k / A_curr* alpha_next.S; 
        dualvars_curr.Z    = A_pre/A_curr * dualvars_k.Z       + a_k / A_curr* alpha_next.Z; 
        dualvars_curr.v    = A_pre/A_curr * dualvars_k.v       + a_k / A_curr* alpha_next.v; 
    end
end
