function [model] = Simple_complex_classifier_iter_1(learningparams, data, idx) 
    data_train_x = data.X(:,idx);
    data_train_y = data.Y(idx);
    data_train_noisy  = data.noisy(idx);
    
    [w2, w_o2] = APG_nonconvex(learningparams, data_train_x, data_train_y, data_train_noisy);
    Options   = set_options();
    [K, K_o ] = get_two_KernelMatrix(data_train_x, learningparams.KOptions);
    excessive_checks = false;
    n_l       = numel(data_train_y);
    Y         = diag(data_train_y);
    n_u       = 0;
    n_S       = n_l;
    n         = n_S;     % size of data 
    nap       = n;
    nSDP      = nap + 1;
    initL     = 1:n_S;
    unlabeled = [];
    extndind  = [];
    c_o       = learningparams.c_o;
    lo        = [zeros(n,1);-ones(numel(extndind),1)];
    up        = ones(numel(nap),1);

    gamma     = ones(n_S,1);
    gamma_un  = 1;%min(n_l/n_u , 1);
    gamma(unlabeled) = gamma_un;
    Gamma     = diag(gamma);
    lambda_o     = learningparams.lambda_o;    
    lambda       = learningparams.lambda;
%%  Define YALMIP Variables        
    p            = sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
    p_pos        = sdpvar(n,1);
    p_neg        = sdpvar(n,1);
    w_o          = sdpvar(n,1);          % For w_o function 
    w            = sdpvar(n,1);
    H            = sdpvar(nSDP,nSDP);
    h            = sdpvar(nap,1);
    beta_p       = sdpvar(nap,1);        % Lagrange for alpha upper bound
    eta_p        = sdpvar(nap,1);        % Lagrange for alpha lower bound
    t            = sdpvar(1,1);          %     
    KVMatrix     = sdpvar(nSDP,nSDP);
    g_D_pos      = sdpvar(n_S,1);
    g_D_neg      = sdpvar(n_S,1);
    u            = [reshape(H,nSDP*nSDP,1);p;g_D_neg]; %g_D_pos is not stored because diag(H)=g_D_pos + g_D_neg; so it can be computed 
    [ wg, w_og] = simple_complex_gasearch(data_train_y, K, K_o, lambda, lambda_o, learningparams.cp);
%%  Define Problem, Constraints and Objective  
    [solve] = doOptimize(Options);
    if solve.solproblem == 0 
        %% Compute \alpha_{opt} using Saddle point problem formulation for comparison with Convex formulation    
        G_p                                = proj_sdp(solve.opt_vars.H(1:n,1:n),n);
        [saddleobj, alpha_opt]             = compSaddleResult(solve.x_opt, solve.opt_vars.H, solve.opt_vars.beta_p, solve.opt_vars.eta_p);
        
        %[solve_iter]   = doOptimize_alternating(Options);
        [solve_iter]   = doOptimize_alternating_homotopy(Options, value(w_o));
        objhomotopy    = objective_func([solve.w;solve.w_o]');
        objfind        = objective_func([value(Y*alpha_opt);value(w_o)]');
        objga          = objective_func([wg;w_og]');
        dnorm          = norm(abs(solve_iter.w)-alpha_opt)/norm(alpha_opt);
        dnorm2         = norm(solve_iter.w_o-solve.x_opt.w_obeta)/norm(solve.x_opt.w_obeta);
        
        ALresult.active= false;
        % set learning parameters 
        model.trainx   = data_train_x;  
        model.trainy   = data_train_y;
        model.n        = numel(data_train_y);
        model.H        = solve.opt_vars.H;
        model.beta_p   = solve.opt_vars.beta_p;
        model.eta_p    = solve.opt_vars.eta_p;
        model.alpha    = alpha_opt;
        model.w_obeta  = solve.x_opt.w_obeta;
        model.x_opt    = solve.x_opt;
        model.obj_opt  = solve.cObjectivev;
        
        model.w_oxT_i  = solve.x_opt.w_obeta'*K_o;
        model.p        = solve.opt_vars.p;
        model.data_train_noisy = data_train_noisy;
        n_noisy        = sum(data_train_noisy);
        
        cmp_noisy      = false(1,n);
        cmp_noisy(k_mostlargest(model.p, n_noisy)) = true; % assume we know the number of noisy instances in advance. 
        noise_rec      = cmp_noisy.*data_train_noisy;
        model.noise_detect_acc = 100*sum(noise_rec)/n_noisy;
        model.maxnoise_p  = max(model.p(cmp_noisy));                       % maximum p for noisy instances
        model.avg_noise_p = sum(model.p(cmp_noisy))/n_noisy;               % Average p for noisy recognized noisy instances.
        model.avg_nonnoise_p = sum(model.p(~cmp_noisy))/(n-n_noisy);       % Average p for non-noisy instances
        abs_wox        = abs(model.w_oxT_i)';
        normdiffp      = norm(abs_wox-solve.opt_vars.p);
        ht             = 1-solve.opt_vars.p;
        model.h        = ht;
        model.name     = 'Simple-Complex_2';
    else
        model = struct([]);
    end
    function yval = objective_func(x)
            Y   = diag(data_train_y);
            w   = x(1:n)';
            w_o = x(n+1:2*n)';
            yval   = sum((1-K_o*Y*w_o).*max(1-K*Y*w,0)) + lambda/2*w'*K*w + lambda_o/2*w_o'*K*w_o + learningparams.cp*sum(abs(K_o*w_o));
    end
    function [solve] = doOptimize_alternating(Options)
        global cnstDefs
        tol          = 10^-4;
        Yl           = data_train_y;
        converged    = false;
        l_o_star     = rand(n,1);
        while ~converged 
            [solve, w_value]      = optimize_w(l_o_star);
            if solve.problem~=0
               solve.cObjectivev = -Inf;
               return;
            end
            l_star = max(1-diag(Yl)*K*w_value, 0 );
            [solve, w_o_value]    = optimize_w_o(l_star);
            if solve.problem~=0
               solve.cObjectivev = -Inf;
               return;
            end
            l_o_st_pre= l_o_star;
            l_o_star  = max(1-diag(Yl)*K_o*w_o_value, 0 );
            if norm(l_o_star-l_o_st_pre) < tol
               converged = true;
            end
        end
        if solve.problem == 0    
            solve.w    = value(w);
            solve.w_o  = value(w_o);
            solve.p    = value(p);
            solve.p_pos= value(p_pos);
            solve.p_neg= value(p_neg);
            solve.cObjectivev = solve.obj;
        else
            solve.cObjectivev = -Inf;
            assert(solproblem~=0,'Problem is not solved in function doOptimize in module ActiveDConvexRelaxOutlier');
        end
    end
    function [solve] = doOptimize_alternating_homotopy(Options, w_o_init)
        global cnstDefs
        tol          = 10^-1;
        lambda_o_hp  = lambda;
        rho_w        = 1;
        rho_w_o      = 1;
        max_iter     = 100;
        step_lamb    = (lambda-lambda_o)/100;
        Yl           = data_train_y;
        w_o_pre      = w_o_init;
        w_pre        = rand(n,1);
        mu_c         = 0.9;
        cp           = learningparams.cp;
        while lambda_o_hp >= lambda_o
            l_o_star  = 1-diag(Yl)*K_o*w_o_pre;
            [solve, w_value]      = optimize_w(l_o_star,w_pre, rho_w);
            if solve.problem~=0
               solve.cObjectivev = -Inf;
               return;
            end
            l_star = max(1-diag(Yl)*K*w_value, 0 );
            [solve, w_o_value]    = optimize_w_o(l_star, w_o_pre, rho_w_o, cp,lambda_o_hp);
            if solve.problem~=0
               solve.cObjectivev = -Inf;
               return;
            end
            if norm(w_value-w_pre)+norm(w_o_value-w_o_pre) < tol
                lambda_o_hp = lambda_o_hp - step_lamb;
            end
            w_pre  = w_value;
            w_o_pre= w_o_value;
        end
        if solve.problem == 0    
            solve.w    = value(w);
            solve.w_o  = value(w_o);
            solve.p    = value(p);
            solve.p_pos= value(p_pos);
            solve.p_neg= value(p_neg);
            solve.cObjectivev = solve.obj;
        else
            solve.cObjectivev = -Inf;
            assert(solproblem~=0,'Problem is not solved in function doOptimize in module ActiveDConvexRelaxOutlier');
        end
    end
    function [sol, w_value]      = optimize_w(l_o_star,w_pre,rho_w)
        Yl           = data_train_y;
        cConstraint  = [];
        cObjective   = sum(l_o_star.*max(1-diag(Yl)*K*w,0))+lambda*w'*K*w/2+rho_w/2*norm(w-w_pre)^2;
        sol          = optimize(cConstraint, cObjective); 
        w_value      = value(w);
        sol.obj      = value(cObjective);
    end
    function [sol, w_o_value]    = optimize_w_o(l_star,w_o_pre, rho_w_o,cp,lambda_o_hp)
        Yl           = data_train_y;
        cConstraint  = [];
        cConstraint  = [cConstraint,  K_o*w_o == p_pos - p_neg];
        cConstraint  = [cConstraint,  p_pos >=0, p_neg>=0];
        cConstraint  = [cConstraint,  p == p_pos + p_neg ];
        cConstraint  = [cConstraint,  p<=1 ];
        cObjective   = sum(l_star.*(1-diag(Yl)*K*w_o))+lambda_o_hp*w_o'*K_o*w_o/2 + cp*sum(p)+rho_w_o/2*norm(w_o-w_o_pre)^2;
        sol          = optimize(cConstraint, cObjective);
        w_o_value    = value(w_o);
        sol.obj      = value(cObjective);
    end
function [solve] = doOptimize(Options)
        global cnstDefs
        [cConstraint, cObjective] = constraintType(Options);
        opts = sdpsettings('verbose', cnstDefs.solver_verbose);
        sol = optimize(cConstraint,cObjective, opts);
        [primalfeas, dualfeas] = check(cConstraint);
        solve.solproblem = sol.problem;
        if solve.solproblem == 0    
            [solve.x_opt, solve.opt_vars] = extractOptResults(cObjective, learningparams, H, w_o, p, h, g_D_pos, g_D_neg, beta_p, eta_p, Options);
            solve.cObjectivev = value(cObjective);
        else
            solve.cObjectivev = -Inf;
            assert(solproblem~=0,'Problem is not solved in function doOptimize in module ActiveDConvexRelaxOutlier');
        end
    end
    function [cConstraint, cObjective]    = constraintType(options)
        Yl           = data_train_y;
        
        M1           = diag([ones(n_S,1);zeros(nap-n_S,1)]);
        M2           = diag([zeros(n_S,1);ones(nap-n_S,1)]);
        MR           = learningparams.lambda_alpha_D_part*M1 + learningparams.lambda_alpha_Q_part*M2;
        
        KVMatrix     = [K.*H(1:nap,1:nap)+ MR ,h+eta_p-beta_p;(h+eta_p-beta_p)',2*t/lambda];        
        %[cConstraint ] = get_labelequivalence_ineq(G_plus(1:n_S,nap+1), n_S, initL, unlabeled);
        cConstraint  = [];
        cConstraint  = [cConstraint, beta_p>=0,      eta_p>=0];
        cConstr1     = [(H>=0):'SDP_H',   (KVMatrix>=0):'SDP_KH'];
        cConstraint  = [cConstraint, cConstr1 ];

        cConstraint  = [cConstraint, (g_D_pos>=0):'h_+',   (g_D_neg>=0):'h_-'];
        cConstraint  = [cConstraint, (h(1:n_S)            == g_D_pos(1:n_S) + g_D_neg(1:n_S)):'abs_h'];
        cConstraint  = [cConstraint, (H(1:n_S,nap+1)      == g_D_pos(1:n_S) - g_D_neg(1:n_S)):'value_h'];
        
        cConstraint  = [cConstraint, (g_D_pos(initL) - g_D_neg(initL) == c_o*Yl'-K_o(initL,:)*w_o):'label_h'];
        cConstraint  = [cConstraint, (H(nap+1,nap+1)      == 1 ):'last_h'];
        cConstraint  = [cConstraint, (diag(H(initL,initL))== h(initL)):'diag_h'];
        cConstraint  = [cConstraint, (h                   == Gamma *(c_o*ones(n_S,1)-Yl'.*(K_o*w_o))):'diagwo_h'];
        cConstraint  = [cConstraint,  K_o*w_o == p_pos - p_neg];
        cConstraint  = [cConstraint,  p_pos >=0, p_neg>=0];
        cConstraint  = [cConstraint,  p == p_pos + p_neg ];
        %cConstraint  = [cConstraint, 0<= p <=c_o ]; 
        
        if options.addsumpConstraint
           if learningparams.label_outlier_seperate_deal== false
              cConstraint=[cConstraint,sum(p)<=n_o*c_o]; 
           else
              cConstraint=[cConstraint,sum(p(initL))<=c_o*n_l*lnoiseper/100,sum(p(unlab))<=c_o*n_u*onoiseper/100];
           end
        end
        cObjective = t+lambda_o*w_o'*K_o*w_o/2+sum(beta_p)+sum(eta_p(extndind))+learningparams.ca*sum(h(unlabeled));
        if options.addsumpobjective 
            cObjective = cObjective + learningparams.cp*sum(p);
        end
        if options.addsumqobjective
            cObjective = cObjective + learningparams.cq*sum(q);
        end
    end
    function [saddleobj, alpha_opt]       = compSaddleResult(x_opt, H, eta_pv, beta_pv)
        global cnstDefs
        alphas         = sdpvar(nap,1);
        x_k            = x_opt;
        H              = (H+H')/2;
        H              = proj_sdp(H,nSDP);
        scObjective    = alphas'*l_of_x(x_k, n, nSDP)...
                    -1/(2*learningparams.lambda)*alphas'*(K.*H(1:nap,1:nap))*alphas ...
                    + eta_of_x(x_k,learningparams, unlabeled, n, nSDP) + learningparams.lambda_o/2*x_k.w_obeta'*K_o*x_k.w_obeta;
        opts = sdpsettings('verbose', cnstDefs.solver_verbose);        
        scConstraint   = [alphas>=lo ,alphas<=up];
        ssol           = optimize(scConstraint,-scObjective, opts);
        if ssol.problem==0
            alpha_opt  = value(alphas);
            rhs        = l_of_x(x_opt, n, nSDP)+eta_pv-beta_pv;
            alphav     = lambda*pinv(K.*H(1:nap,1:nap))*rhs;
            objsc      = value(scObjective);            
            saddleobj  = value(scObjective); %f_of_xAndAlpha(x_opt, alpha_opt,learningparams);
        else
            saddleobj      = -111111111111111;
        end 
    end
    function [Options] = set_options()
        % options for making constraints
    Options.constraintForm = 5; Options.addsumpConstraint = false; Options.addsumpobjective = true; Options.addsumqobjective = false;
    Options.diagunlabIneq = false; Options.diagsetQueryIneq = false; 
    Options.useq = false;
    Options.useoperators = false;
    Options.useoperators = false;
    end
end
function [x_opt, opt_vars] = extractOptResults(cObjective, learningparams, H, w_o, p, h, g_D_pos, g_D_neg, beta_p, eta_p, Options)
    %% Select Samples to query from : this code not written for part of unlabeled data are being queried from. 
    % Select All of unlabeled samples for Querying
    opt_vars.H        = value(H);
    opt_vars.p        = value(p);
    opt_vars.w_o      = value(w_o);
    opt_vars.g_D_pos  = value(g_D_pos);
    opt_vars.g_D_neg  = value(g_D_neg);
    opt_vars.beta_p   = value(beta_p);
    opt_vars.eta_p    = value(eta_p);
    opt_vars.h        = value(h);
    st               = 0;
    [x_opt] = x_conv_abs_h(opt_vars.H, opt_vars.p, opt_vars.g_D_neg, opt_vars.w_o ,st);
end