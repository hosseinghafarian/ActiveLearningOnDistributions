function [model] = Simple_complex_classifier_3(learningparams, data, idx) 
    data_train_x = data.X(:,idx);
    data_train_y = data.Y(idx);
    data_train_noisy  = data.noisy(idx);
    
    Options = set_options();
    [K, K_o ] = get_two_KernelMatrix(data_train_x, learningparams.KOptions);
    excessive_checks = false;
    n_l       = numel(data_train_y);
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
%     if learningparams.label_outlier_seperate_deal == false
%        n_o       = learningparams.n_o;
%        lnoiseper = learningparams.lnoiseper;
%        onoiseper = learningparams.onoiseper;
%     else
%        lnoiseper = learningparams.lnoiseper;
%        onoiseper = learningparams.onoiseper;
%     end
    lambda_o     = learningparams.lambda_o;    
    lambda       = learningparams.lambda;
%%  Define YALMIP Variables        
    p            = sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
    w_o          = sdpvar(n,1);          % For w_o function 
    H            = sdpvar(nSDP,nSDP);
    h            = sdpvar(nap,1);
    v            = sdpvar(nap,1);
    beta_p       = sdpvar(nap,1);        % Lagrange for alpha upper bound
    eta_p        = sdpvar(nap,1);        % Lagrange for alpha lower bound
    t            = sdpvar(1,1);          %     
    KVMatrix     = sdpvar(nSDP,nSDP);
    g_D_pos      = sdpvar(n_S,1);
    g_D_neg      = sdpvar(n_S,1);
    u            = [reshape(H,nSDP*nSDP,1);p;g_D_neg]; %g_D_pos is not stored because diag(H)=g_D_pos + g_D_neg; so it can be computed 
%%  Define Problem, Constraints and Objective  
    [solve] = doOptimize(Options);
    if solve.solproblem == 0 
        %% Compute \alpha_{opt} using Saddle point problem formulation for comparison with Convex formulation    
        G_p                                = proj_sdp(solve.opt_vars.H(1:n,1:n),n);
        [saddleobj, alpha_opt]             = compSaddleResult(solve.x_opt, solve.opt_vars.H, solve.opt_vars.beta_p, solve.opt_vars.eta_p);
        if excessive_checks 
%% Commented             showWhatsdoing                 = 'Testing equality of Convex and Saddle Formulations....';
%             disp(showWhatsdoing);
%             assert(abs(saddleobj-cObjectivev)/abs(cObjectivev) < tolobj,'Saddle Problem Objective doesnot match that of Convex Problem')
%             [ft,g_x,g_alpha]                   = f_xAlpha_grad(x_opt,alpha_opt,learningparams);
%             assert(abs(ft-cObjectivev)/abs(cObjectivev) < tolobj,'Objective function value with optimal variables doesnot match that of Convex Problem');
%             %% Recompute x_opt, using Computed \alpha_{opt}
%             [sdpObjective, x_opt2,G_plusv2,qv2,pv2,beta_pv2,eta_pv2,misc2] = solve_using_operators_and_alpha(learningparams, Model, Options, WMST_beforeStart, alpha_opt);
%             diffM          = G_plusv2-G_plusv;
%             normdiffM      = norm(diffM);
%             assert(abs(sdpObjective-cObjectivev)/abs(cObjectivev)<tolobj,'SDP problem objective doesnot match that of Convex Problem');
%             [equalitydiff, eq, Inequality, Ineq]             = ConstraintsCheck(x_opt2, learningparams);
%             assert(equalitydiff < toleq,'Equality constraints are not satisfied');
%             assert(Inequality   < toleq,'Inequality constraints are not satisfied');
%             ALresult.alphav    = alpha_opt;
%             ALresult.saddleobj = saddleobj;
        end
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
        
        model.w_oxT_i  = solve.x_opt.w_obeta'*diag(data_train_y)*K_o;
        model.p        = solve.opt_vars.p;
        pt             = model.p;

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
        model.name     = 'Simple-Complex_1';
    else
        model = struct([]);
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
        
        KVMatrix     = [K.*H(1:nap,1:nap)+ MR ,v+eta_p-beta_p;(v+eta_p-beta_p)',2*t/lambda];        
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
        cConstraint  = [cConstraint, (v                   >= Gamma *(c_o*ones(n_S,1)-2*c_o*Yl'.*(K_o*w_o))):'v_2'];
        cConstraint  = [cConstraint, (v                   >=0)];
        cConstraint  = [cConstraint, -p<= K_o*w_o <=p];
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
function  [sdpObjective, x_opt2,G_plusv2,qv2,pv2,beta_pv2,eta_pv2,misc2]= solve_using_operators_and_alpha(learningparams, Model, Options, WMST_beforeStart, alpha_opt)
    global cnstData
    global WMST_variables
    n_l       = cnstData.n_l;
    %% Select Samples to query from : this code not written for part of unlabeled data are being queried from. 
    % Select All of unlabeled samples for Querying
    n        = cnstData.n_S;     % size of data 
    nap      = cnstData.nap;
    extndind = cnstData.extendInd;
    assert(~isempty(extndind));
    yalmip('clear');
    p       = sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
    w_o     = sdpvar(n,1);          % For w_o function 
    G_plus  = sdpvar(nap+1,nap+1);  
    q       = sdpvar(cnstData.n_q,1);        
    beta_p  = sdpvar(nap,1);        % Lagrange for alpha upper bound
    eta_p   = sdpvar(nap,1);        % Lagrange for alpha lower bound
    t       = sdpvar(1,1);          %     
    KVMatrix= sdpvar(nap+1,nap+1);
    h       = sdpvar(nap,1);
    g_D_pos = sdpvar(n_S,1);
    g_D_neg = sdpvar(n_S,1);
    u       = [reshape(G_plus,cnstData.nSDP*cnstData.nSDP,1);p;g_D_neg];
%             [scoperators]  = getConstraints3(learningparams);
    [scoperators, y_EC, y_EV, y_IC, y_IV,x_st_IC, x_st_IV]                = getConstraints5(learningparams, false, WMST_beforeStart, WMST_variables, Model.model);   
    s1             = sdpvar(scoperators.n_AIC,1);
    s2             = sdpvar(scoperators.n_AIV,1);
    s              = [s1;s2];
    [sadcConstraint, csdpObjective, x_varsdp]       = operconstrSaddleType(scoperators, learningparams, G_plus, p, w_o, q, r, beta_p, eta_p, g_D, h, rl, u, s1, s2, s, alpha_opt, Options.useoperators,Options.addsumpConstraint, Options.addsumpobjective, Options.addsumqobjective,Options.useq, Options.diagunlabIneq,Options.diagsetQueryIneq);
    sol_saddle     = optimize(sadcConstraint,csdpObjective);
    sdpObjective   = value(csdpObjective);
    obj_opt        = sdpObjective;
    [x_opt2,G_plusv2,qv2,pv2,beta_pv2,eta_pv2,misc2] = extractOptResults(csdpObjective, learningparams, q, G_plus, p, w_o, g_D, h, r, rl, beta_p, eta_p, s1, s2, Options); 
end
function [queryind,Y_set]             = getQueryAndPredLabel(G_plusv, pv, qv, alpha_opt)
            %% Discussion about using use of qyu for obtaining q
            %y_ui may be less than 1 or greater than -1. In these cases,
            %q value may mislead us. 
            % I think the best value is this value which is the nearest
            % value to y_ui*(1-q_i): may be that's because r is not exactly
            % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
            % [-1,1],but we must assume it is {-1,1}     
            %ra must be equal to absoulte value of qyu but in pratice the
            %value is completely different due to relaxation of y_ui to [-1,1] 
            %sa    =value(s);
%             if formp==1
%                  qresult=1-abs(qyu);
%                  qresult=qresult.*(1-pv(n_l+1:n));
%             else
%                  qresult=qv;    
%             end
            %% Obtaining results
            Y_set             = sign(value(G_plusv(1:n,nap+1))); 
            qyu               = G_plusv(setunlab,nap+1);
            % find largest p, (they are the noisiest)
            separate          = true;
            [noisysamplesInd] = get_noisy_sample_ind(pv, n, lnoiseper, onoiseper, separate);
            % retain only unlabeled samples
            isunlabelednoisy  = ismember(noisysamplesInd,unlabeled);
            noisysamplesInd   = noisysamplesInd(isunlabelednoisy);
            alsubtype         = 8;
            qresult           = zeros(n,1);
            if alsubtype==1
                qresult(samples_toQuery_from) = qv;
            elseif alsubtype==2
                qresult(samples_toQuery_from) = qv;
                qresult = qresult.*(1-pv);
            elseif alsubtype==3
                qresult(samples_toQuery_from) = 1-abs(qyu);
                qresult = qresult.*(1-pv);                
            elseif alsubtype==4
                qresult(samples_toQuery_from) = 1-abs(qyu);
                qresult = qresult.*(1-pv);
                qresult(noisysamplesInd) = 0;          % TODO: it must be checked that p_i for noisydata are significantly larger than others,
                                                       % otherwise if for example all of them are zero it has no meaning
            elseif alsubtype==5
                qresult = qv.*(1-pv);
                qresult(noisysamplesInd) = 0; 
            elseif alsubtype==6
                qresult(samples_toQuery_from) = qv;    
                qresult(noisysamplesInd) = 0;
            elseif alsubtype==7
                f_inst  = 1/lambda*cnstData.K*(Y_set.*alpha_opt(1:n_S));
                absf_in = abs(f_inst);
                max_absf= max(absf_in);
                qresult = max_absf-absf_in;
            elseif alsubtype==8
                f_inst  = 1/lambda*cnstData.K(samples_toQuery_from,initL)*(Y_set(initL).*alpha_opt(initL));
                absf_in = abs(f_inst);
                max_absf= max(absf_in);
                qresult(samples_toQuery_from)= max_absf - absf_in;
            end
            [tq, success] = k_mostlargest_from(qresult, batchSize, samples_toQuery_from);
            queryind = tq;%unlabeled(tq); 
end
function [cConstraint ] = get_labelequivalence_ineq(rY, n, initL, unlabeled)
    labeled_one          = zeros(n,1);
    labeled_one(initL)   = 1;
    n_lab_one            = sum(labeled_one);
    unlab_one            = zeros(n,1);
    unlab_one(unlabeled) = 1;
    n_unlab_one          = sum(unlab_one);
    unb_lab_eps          = 0.1;
%     if numel(initL) >2 && numel(initL)<8
        cConstraint              = [-unb_lab_eps <= 1/n_lab_one*(labeled_one'*rY)-1/n_unlab_one*(unlab_one'*rY),...
                                    1/n_lab_one*(labeled_one'*rY)-1/n_unlab_one*(unlab_one'*rY)<= unb_lab_eps ];
%     else
%         cConstraint              = [];
%     end
end
function [checkRes ] = check_labelequivalence_ineq(rY, n, initL, unlabeled)
    labeled_one          = zeros(n,1);
    labeled_one(initL)   = 1;
    n_lab_one            = sum(labeled_one);
    unlab_one            = zeros(n,1);
    unlab_one(unlabeled) = 1;
    n_unlab_one          = sum(unlab_one);
    unb_lab_eps          = 0.1;
    checkRes             = false;
%     if numel(initL) <=2 || numel(initL)>=8
%        checkRes = true;
%        return
%     end
    if -unb_lab_eps <= 1/n_lab_one*(labeled_one'*rY)-1/n_unlab_one*(unlab_one'*rY)
        if 1/n_lab_one*(labeled_one'*rY)-1/n_unlab_one*(unlab_one'*rY)<= unb_lab_eps
            checkRes = true;
            return;
        end
    end    
end
function [cConstraint, cObjective]    = operatorconstraintType(scoperators,useoperators,addsumpConstraint, addsumpobjective, addsumqobjective,useq, diagunlabIneq,diagsetQueryIneq)
    %KVMatrix   = [cnstData.KE.*G_plus(1:nap,1:nap) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
    KVMatrix     = [cnstData.KE.*G_plus(1:nap,1:nap) ,h+eta_p-beta_p;(h+eta_p-beta_p)',2*t/lambda];


    %[cConstraint ] = get_labelequivalence_ineq(G_plus(1:n_S,nap+1), n_S, initL, unlabeled);
    cConstraint= [];
    cConstraint  = [cConstraint, beta_p>=0 ];
    cConstraint  = [cConstraint, eta_p>=0];
    cConstraint  = [cConstraint, G_plus>=0];
    cConstraint  = [cConstraint, KVMatrix>=0]; %nonnegativity constraints

    pcConstraint = scoperators.A_EC*u ==scoperators.b_EC;
    pcConstraint2= scoperators.A_EV*u  + scoperators.B_EV*w_o == scoperators.b_EV;
    pcConstraint3= [scoperators.A_IC*u==s1,s1<= scoperators.s_IC];
    pcConstraint4= [scoperators.A_IV*u + scoperators.B_IV*w_o==s2,s2<=scoperators.s_IV];

    cConstraint  = [cConstraint, pcConstraint, pcConstraint2, pcConstraint3,pcConstraint4];


    %% Attention: for times when all of unlabeled data are not queried the following code must be checked
    %cConstraint= [cConstraint, 0<=G_plus(extndind,nap+1),G_plus(extndind,nap+1)<=1];% constraints on q%%implict 
    if diagsetQueryIneq 
        %cConstraint= [cConstraint, diag(G_plus(extndind,extndind))<=G_plus(extndind,nap+1)  ];
    else
        %cConstraint= [cConstraint, diag(G_plus(extndind,extndind))==G_plus(extndind,nap+1)  ];
    end
    %cConstraint= [cConstraint, G_plus(initL,nap+1)==rl];

    cConstraint= [cConstraint, h(initL)   ==diag(G_plus(initL,initL))];%1-p(initL) ];
    cConstraint= [cConstraint, h(extndind)==zeros(cnstData.n_q,1)];
    cConstraint= [cConstraint, h(setunlab)==diag(G_plus(setunlab,setunlab))];%diag(G_plus(setunlab,setunlab))];%
    %cConstraint= [cConstraint, -p<=cnstData.K_o*w_o<=p];
    %cConstraint= [cConstraint, p<=1 ];
    %cConstraint= [cConstraint, rl==Yl-cnstData.K_o(initL,:)*w_o];
    %cConstraint= [cConstraint, diag(G_plus(setunlab,setunlab))>=G_plus(setunlab,nap+1), diag(G_plus(setunlab,setunlab))>=-G_plus(setunlab,nap+1)]; % r is for absolute value of y_u.*(1-pu).*(1-q)

    if addsumpConstraint
        %cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
    end

    cObjective = t+lambda_o*w_o'*cnstData.K_o*w_o/2+sum(beta_p)+sum(eta_p(extndind))+learningparams.ca*sum(diag(G_plus(setunlab,setunlab)-hbar));
    if addsumpobjective 
        cObjective = cObjective + learningparams.cp*sum(p);
    end
    if addsumqobjective
        cObjective = cObjective + learningparams.cq*sum(q);
    end
end
%% function [cConstraint, cObjective, x_varsdp]    = operconstrSaddleType(scoperators, learningparams, G_plus, p, w_o, q, r, beta_p, eta_p, g_D, h, rl, u, s1, s2, s, alphav, useoperators,addsumpConstraint, addsumpobjective, addsumqobjective,useq, diagunlabIneq,diagsetQueryIneq)
%         cConstraint  = G_plus>=0;
%         
%         [cConstraint ] = get_labelequivalence_ineq(cConstraint, G_plus, nap, initL, unlabeled);
%         
%         x_varsdp     = x_conv(G_plus,p,w_o,[s1;s2]);        
%         pcConstraint = scoperators.A_EC*x_varsdp.u==scoperators.b_EC;
%         pcConstraint2= scoperators.A_EV*x_varsdp.u+ scoperators.B_EV*x_varsdp.w_obeta == scoperators.b_EV;
%         pcConstraint3= [scoperators.A_IC*x_varsdp.u==s1,s1<=scoperators.s_IC];
%         pcConstraint4= [scoperators.A_IV*x_varsdp.u+scoperators.B_IV*x_varsdp.w_obeta == s2,s2<=scoperators.s_IV];
%         cConstraint  = [cConstraint, pcConstraint, pcConstraint2, pcConstraint3,pcConstraint4];
%         %% Attention: for times when all of unlabeled data are not queried the following code must be checked
%         if diagsetQueryIneq 
%             %cConstraint= [cConstraint, diag(G_plus(extndind,extndind))<=G_plus(extndind,nap+1)  ];
%         else
%             %cConstraint= [cConstraint, diag(G_plus(extndind,extndind))==G_plus(extndind,nap+1)  ];
%         end
%         if addsumpConstraint
% 
%         end
%         typ         = 2;
%         cObjective = f_of_xAndAlpha(x_varsdp, alphav,learningparams,typ);
%         if addsumpobjective 
% %            cObjective = cObjective + learningparams.cp*sum(p);
%         end
%         if addsumqobjective
% %            cObjective = cObjective + learningparams.cq*sum(q);
%         end
% end