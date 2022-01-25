function [ALresult, model_set, Y_set]= ActiveDCvxRlxOt_abs_h_abs_p(WMST_beforeStart, Model, learningparams, xTrain, yTrain)
        global cnstData
        global WMST_variables
        ALresult.active  = true;
        excessive_checks = false;
        initL     = cnstData.initL(cnstData.initL>0);
        unlabeled = cnstData.unlabeled;
        n_l       = cnstData.n_l;
        n_u       = cnstData.n_u;
        n_S       = cnstData.n_S;
        batchSize = cnstData.batchSize;
        gamma     = ones(n_S,1);
        gamma_un  = min(n_l/n_u , 1);
        gamma(unlabeled) = gamma_un;
        % gamma(unlabeled) = n_l/n_u; ÃÊ«» ŒÊ»? œ«œ,  « ‰?„Â œ«œÂ Â«, »⁄œ
        % 
        %gamma(unlabeled) = 1/n_u ;
        Gamma     = diag(gamma);
        if cnstData.label_outlier_seperate_deal == false
           n_o       = cnstData.n_o;
        else
           lnoiseper = cnstData.lnoiseper;
           onoiseper = cnstData.onoiseper;
        end
        toleq     = 10^-3;
        tolobj    = 10^-3;
        hbar      = cnstData.hbar;
        lambda_o  = learningparams.lambda_o;    
        lambda    = learningparams.lambda;
        %% Select Samples to query from : this code not written for part of unlabeled data are being queried from. 
        % Select All of unlabeled samples for Querying
        n        = cnstData.n_S;     % size of data 
        nap      = cnstData.nap;
        setunlab = cnstData.unlabeled;
        extndind = cnstData.extendInd;
        assert(~isempty(extndind));
        query    = cnstData.query;
        unlabNotQ= setdiff(setunlab,query);
        samples_toQuery_from = cnstData.unlabeled; % Select the set to query from     
        fromunlabtoQuery     = ismember(cnstData.unlabeled, samples_toQuery_from);  % this variable determines which unlabeled samples are queried
%%      Define YALMIP Variables        
        p       = sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
        p_pos   = sdpvar(n,1);          % For postive part of Outlier function w_o^T\phi(x_i)
        p_neg   = sdpvar(n,1);          % For negative part of Outlier function w_o^T\phi(x_i)
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
        u       = [reshape(G_plus,cnstData.nSDP*cnstData.nSDP,1);p;g_D_neg]; %g_D_pos is not stored because diag(H)=g_D_pos + g_D_neg; so it can be computed 
%%      Define Problem, Constraints and Objective  
        % options for making constraints
        Options.constraintForm = 5; Options.addsumpConstraint = false; Options.addsumpobjective = true; Options.addsumqobjective = false;
        Options.diagunlabIneq = false; Options.diagsetQueryIneq = false; 
        Options.useq = false;
        Options.useoperators = false;
        if Options.useoperators 
            [scoperators, y_EC, y_EV, y_IC, y_IV,x_st_IC, x_st_IV]                = getConstraints5(learningparams, false, WMST_beforeStart, WMST_variables, Model.model);   
            %[scoperators]  = getConstraints3(learningparams);
            s1             = sdpvar(scoperators.n_AIC,1);
            s2             = sdpvar(scoperators.n_AIV,1);
            s              = [s1;s2];
             [solproblem, cObjectivev, x_opt, G_plusv, qv, pv,beta_pv,eta_pv, misc ] = doOptimize(Options);
        else
            Options.useoperators = false;
            [solproblem, cObjectivev, x_opt, G_plusv, qv, pv, p_posv, p_negv, beta_pv,eta_pv, misc ] = doOptimize(Options);
        end
        if solproblem == 0 
            %% Compute \alpha_{opt} using Saddle point problem formulation for comparison with Convex formulation    
            G_p                                = proj_sdp(G_plusv(1:nap,1:nap),nap);
            [saddleobj,alpha_opt]              = compSaddleResult(x_opt, G_p, beta_pv, eta_pv);
            %% Compute Query indices
            [queryind,Y_set]                   = getQueryAndPredLabel(G_plusv, pv, qv, alpha_opt);
            % set active learning results
            ALresult.q                         = zeros(n,1);
            ALresult.q                         = qresult;
            ALresult.samples_toQuery_from      = samples_toQuery_from;
            ALresult.queryind                  = queryind;% samples_toQuery_from(tq);
            ALresult.qBatch                    = zeros(n_u,1);
            ALresult.qBatch(ALresult.queryind) = 1;
            if excessive_checks 
                
                showWhatsdoing                     = 'Testing equality of Convex and Saddle Formulations....';
                disp(showWhatsdoing);
                assert(abs(saddleobj-cObjectivev)/abs(cObjectivev) < tolobj,'Saddle Problem Objective doesnot match that of Convex Problem')
                [ft,g_x,g_alpha]                   = f_xAlpha_grad(x_opt,alpha_opt,learningparams);
                assert(abs(ft-cObjectivev)/abs(cObjectivev) < tolobj,'Objective function value with optimal variables doesnot match that of Convex Problem');
                %% Recompute x_opt, using Computed \alpha_{opt}
                [sdpObjective, x_opt2,G_plusv2,qv2,pv2,beta_pv2,eta_pv2,misc2] = solve_using_operators_and_alpha(learningparams, Model, Options, WMST_beforeStart, alpha_opt);
                diffM          = G_plusv2-G_plusv;
                normdiffM      = norm(diffM);
                assert(abs(sdpObjective-cObjectivev)/abs(cObjectivev)<tolobj,'SDP problem objective doesnot match that of Convex Problem');
                [equalitydiff, eq, Inequality, Ineq]             = ConstraintsCheck(x_opt2, learningparams);
                assert(equalitydiff < toleq,'Equality constraints are not satisfied');
                assert(Inequality   < toleq,'Inequality constraints are not satisfied');
                ALresult.alphav    = alpha_opt;
                ALresult.saddleobj = saddleobj;
            end
%             fxoptalpha2  = value(f_of_xAndAlpha(x_varsdp,alpha_opt,learningparams));
%             fxoptalpha   = f_of_xAndAlpha(x_opt2,alpha_opt,learningparams);            
%             assert(abs(fxoptalpha-fxoptalpha2)/abs(fxoptalpha2)<tolobj,'Objective function values doesnot match');
            % checking constraints
            % set learning parameters 
            model_set.beta_p   = beta_pv;
            model_set.eta_p    = eta_pv;
            model_set.w_oxT_i  = x_opt.w_obeta'*cnstData.K_o;
            model_set.G        = G_plusv;
            model_set.alpha_pv = alpha_opt;
            model_set.p_neg    = p_negv;
            model_set.p_pos    = p_posv;
            model_set.p        = pv;
            model_set.noisySamplesInd = noisysamplesInd;
            query_subset       = cnstData.query;
            ht                 = 1-pv;
            ht(query_subset)   = ht(query_subset)-qv;
            model_set.h        = ht;
            model_set.w_obeta  = x_opt.w_obeta;
            model_set.x_opt    = x_opt;
            model_set.obj_opt  = cObjectivev;
    end
    function [cConstraint, cObjective]    = constraintType(addsumpConstraint, addsumpobjective, addsumqobjective,useq, diagunlabIneq,diagsetQueryIneq)
        
        initLStart_notnoisy = cnstData.initLStart_notnoisy;
        initLStart          = cnstData.initLStart;
        
        KVMatrix   = [cnstData.KE.*G_plus(1:nap,1:nap) ,h+eta_p-beta_p;(h+eta_p-beta_p)',2*t/lambda];
        %[cConstraint ] = get_labelequivalence_ineq(G_plus(1:n_S,nap+1), n_S, initL, unlabeled);
        cConstraint= [];
        cConstraint= [cConstraint, beta_p>=0];
        cConstraint= [cConstraint, eta_p>=0];
        cConstraint= [cConstraint, G_plus>=0];
        cConstraint= [cConstraint, KVMatrix>=0]; %nonnegativity constraints
        if useq 
            cConstraint= [cConstraint, sum(q)==batchSize ];
            cConstraint= [cConstraint, 0<=q,q<=1];% constraints on q%%implict 
            % The following two constraints are replaced for r+q+p(unlab)==1
            % For times when some of unlabeled data are not in query,i.e., we
            % didn't query any unlabeled data 
            cConstraint= [cConstraint, h(query)+gamma(query).*q+gamma(query).*p(query)==gamma(query)]; 
            if numel(unlabNotQ)>0 
                cConstraint= [cConstraint, h(unlabNotQ)+gamma(unlabNotQ).*p(unlabNotQ)==gamma(unlabNotQ)]; 
            end
            cConstraint= [cConstraint, G_plus(extndind,nap+1)==q ];
            if diagsetQueryIneq 
                cConstraint= [cConstraint, diag(G_plus(extndind,extndind))<=q       ];
            else
                cConstraint= [cConstraint, diag(G_plus(extndind,extndind))==q       ];
            end
        else %% Attention: for times when all of unlabeled data are not queried the following code must be checked.
            try 
               cConstraint= [cConstraint, sum(G_plus(extndind,nap+1))==batchSize ];
            catch 
               warning('something went wrong here in constraintType');
            end
            cConstraint= [cConstraint, 0<=G_plus(extndind,nap+1),G_plus(extndind,nap+1)<=1];% constraints on q%%implict 
            cConstraint= [cConstraint, h(query)+gamma(query).*G_plus(extndind,nap+1)+gamma(query).*p(query)==gamma(query)];
            if diagsetQueryIneq 
                cConstraint= [cConstraint, diag(G_plus(extndind,extndind))<=G_plus(extndind,nap+1)  ];
            else
                cConstraint= [cConstraint, diag(G_plus(extndind,extndind))==G_plus(extndind,nap+1)  ];
            end
        end
        cConstraint= [cConstraint, g_D_pos>=0,  g_D_neg>=0];
        cConstraint= [cConstraint, h(1:n_S)            == g_D_pos(1:n_S) + g_D_neg(1:n_S)];
        cConstraint= [cConstraint, G_plus(1:n_S,nap+1) == g_D_pos(1:n_S) - g_D_neg(1:n_S)];
        Yl         = cnstData.Yl(cnstData.initL(cnstData.initLnozero));
        cConstraint= [cConstraint, g_D_pos(initL) - g_D_neg(initL) == Yl-cnstData.K_o(initL,:)*w_o];
        cConstraint= [cConstraint, G_plus(nap+1,nap+1)==1 ];
        cConstraint= [cConstraint, diag(G_plus(initL,initL))==gamma(initL)-gamma(initL).*p(initL)];
        cConstraint= [cConstraint, p_pos>=0, p_neg>=0];
        cConstraint= [cConstraint, cnstData.K_o*w_o == p_pos-p_neg];
        cConstraint= [cConstraint,                p == p_pos+p_neg];
        %cConstraint= [cConstraint, -p<=cnstData.K_o*w_o<=p];
        cConstraint= [cConstraint, p<=1 ];
        
        if initLStart_notnoisy
           cConstraint = [cConstraint, p(initLStart)==0];
        end

        if addsumpConstraint
           if cnstData.label_outlier_seperate_deal== false
              cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
           else
              cConstraint=[cConstraint,sum(p(initL))<=n_l*lnoiseper/100,sum(p(setunlab))<=n_u*onoiseper/100];
           end
        end
        
        if diagunlabIneq
            cConstraint= [cConstraint, diag(G_plus(setunlab,setunlab))<=h(setunlab) ];
        else
            cConstraint= [cConstraint, diag(G_plus(setunlab,setunlab))==h(setunlab) ];
        end
        cConstraint= [cConstraint, h(initL)   ==diag(G_plus(initL,initL))];%1-p(initL) ];
        cConstraint= [cConstraint, h(extndind)==zeros(cnstData.n_q,1)];
        cConstraint= [cConstraint, h(setunlab)==diag(G_plus(setunlab,setunlab))];%diag(G_plus(setunlab,setunlab))];%        
%        cConstraint= [cConstraint, h(setunlab)>=G_plus(setunlab,nap+1), h(setunlab)>=-G_plus(setunlab,nap+1)]; % r is for absolute value of y_u.*(1-pu).*(1-q)
        cObjective = t+lambda_o*w_o'*cnstData.K_o*w_o/2+sum(beta_p)+sum(eta_p(extndind))+learningparams.ca*sum(h(setunlab)-hbar);
        if addsumpobjective 
            cObjective = cObjective + learningparams.cp*sum(p);
        end
        if addsumqobjective
            cObjective = cObjective + learningparams.cq*sum(q);
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
            tq = k_mostlargest(qresult,batchSize);
            queryind = tq;%unlabeled(tq); 
    end
    function [saddleobj, alpha_opt]       = compSaddleResult(x_opt,G_plusv,eta_pv,beta_pv)
        global cnstDefs
        alphas         = sdpvar(nap,1);
        x_k            = x_opt;
        G_plusv        = (G_plusv+G_plusv')/2;
        G_plusv        = proj_sdp(G_plusv,cnstData.nSDP);
        %scObjective   = -f_of_xAndAlpha(x_opt, alphas,learningparams);
        scObjective    = alphas'*l_of_x(x_k)...
                    -1/(2*learningparams.lambda)*alphas'*(cnstData.KE.*G_plusv(1:nap,1:nap))*alphas ...
                    + eta_of_x(x_k,learningparams) + learningparams.lambda_o/2*x_k.w_obeta'*cnstData.K_o*x_k.w_obeta;
        opts = sdpsettings('verbose', cnstDefs.solver_verbose);        
        scConstraint   = [alphas>=cnstData.lo,alphas<=cnstData.up];
        ssol           = optimize(scConstraint,-scObjective, opts);
        if ssol.problem==0
            alpha_opt  = value(alphas);
            rhs        = l_of_x(x_opt)+eta_pv-beta_pv;
            alphav     = lambda*pinv(cnstData.KE.*G_plusv(1:nap,1:nap))*rhs;
            objv       = -f_of_xAndAlpha(x_opt, alpha_opt,learningparams);
            objsc      = value(scObjective);            
            saddleobj  = value(scObjective); %f_of_xAndAlpha(x_opt, alpha_opt,learningparams);
        else
            saddleobj      = -111111111111111;
        end 
    end
    function [solproblem, cObjectivev, x_opt, G_plusv, qv, pv, p_posv, p_negv, beta_pv,eta_pv, misc ] = doOptimize(Options)
        global cnstDefs
        
        if Options.useoperators
            [cConstraint, cObjective] = operatorconstraintType(scoperators,Options.useoperators, Options.addsumpConstraint, Options.addsumpobjective, Options.addsumqobjective,Options.useq, Options.diagunlabIneq,Options.diagsetQueryIneq);
        else
            [cConstraint, cObjective] = constraintType(Options.addsumpConstraint, Options.addsumpobjective, Options.addsumqobjective,Options.useq, Options.diagunlabIneq,Options.diagsetQueryIneq);
        end
        %% Consider percentage of labeled noise data and unlabeled noise data
        opts = sdpsettings('verbose', cnstDefs.solver_verbose);
        sol = optimize(cConstraint,cObjective, opts);
        [primalfeas, dualfeas] = check(cConstraint);
        solproblem = sol.problem;
        if solproblem == 0
            if ~Options.useoperators
               s1 = 0;
               s2 = 0;
            end    
            [x_opt,G_plusv,qv,pv, p_posv, p_negv,beta_pv,eta_pv,misc] ...
                   = extractOptResults(cObjective, learningparams, q, G_plus, p, p_pos, p_neg, w_o, g_D_pos, g_D_neg, h, beta_p, eta_p, s1, s2, Options);
            cObjectivev = value(cObjective);
            %[checkRes ] = check_labelequivalence_ineq(G_plusv(1:n_S,nap+1), n_S, initL, unlabeled);
            %assert(checkRes,'Balanced labeled Constraints doesnot satisfied');
        else
            assert(solproblem~=0,'Problem is not solved in function doOptimize in module ActiveDConvexRelaxOutlier');
        end
    end
    function [yalvars] = createYalmipVars(scoperators)
        n        = cnstData.n_S;     % size of data 
        nap      = cnstData.nap;
        setunlab = cnstData.unlabeled;
        extndind = cnstData.extendInd;
        query    = cnstData.query;
        unlabNotQ= setdiff(setunlab,query);
        samples_toQuery_from = cnstData.unlabeled; % Select the set to query from     
        fromunlabtoQuery     = ismember(cnstData.unlabeled, samples_toQuery_from);  % this variable determines which unlabeled samples are queried
%%      Define YALMIP Variables        
        yalvars.p       = sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
        yalvars.w_o     = sdpvar(n,1);          % For w_o function 
        yalvars.G_plus  = sdpvar(nap+1,nap+1);  
        yalvars.q       = sdpvar(cnstData.n_q,1);        
        yalvars.r       = sdpvar(n,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 
        yalvars.beta_p  = sdpvar(nap,1);        % Lagrange for alpha upper bound
        yalvars.eta_p   = sdpvar(nap,1);        % Lagrange for alpha lower bound
        yalvars.t       = sdpvar(1,1);          %     
        yalvars.KVMatrix= sdpvar(nap+1,nap+1);
        yalvars.g_D     = sdpvar(nap,1);
        yalvars.h       = sdpvar(nap,1);
        yalvars.rl      = sdpvar(n_l,1);
        
        yalvars.u       = [reshape(G_plus,cnstData.nSDP*cnstData.nSDP,1);p];
        yalvars.s1      = sdpvar(scoperators.n_AIC,1);
        yalvars.s2      = sdpvar(scoperators.n_AIV,1);
        yalvars.s       = [s1;s2]; 
    end
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
function [cConstraint, cObjective, x_varsdp]    = operconstrSaddleType(scoperators, learningparams, G_plus, p, w_o, q, r, beta_p, eta_p, g_D, h, rl, u, s1, s2, s, alphav, useoperators,addsumpConstraint, addsumpobjective, addsumqobjective,useq, diagunlabIneq,diagsetQueryIneq)
        cConstraint  = G_plus>=0;
        
        [cConstraint ] = get_labelequivalence_ineq(cConstraint, G_plus, nap, initL, unlabeled);
        
        x_varsdp     = x_conv(G_plus,p,w_o,[s1;s2]);        
        pcConstraint = scoperators.A_EC*x_varsdp.u==scoperators.b_EC;
        pcConstraint2= scoperators.A_EV*x_varsdp.u+ scoperators.B_EV*x_varsdp.w_obeta == scoperators.b_EV;
        pcConstraint3= [scoperators.A_IC*x_varsdp.u==s1,s1<=scoperators.s_IC];
        pcConstraint4= [scoperators.A_IV*x_varsdp.u+scoperators.B_IV*x_varsdp.w_obeta == s2,s2<=scoperators.s_IV];
        cConstraint  = [cConstraint, pcConstraint, pcConstraint2, pcConstraint3,pcConstraint4];
        %% Attention: for times when all of unlabeled data are not queried the following code must be checked
        if diagsetQueryIneq 
            %cConstraint= [cConstraint, diag(G_plus(extndind,extndind))<=G_plus(extndind,nap+1)  ];
        else
            %cConstraint= [cConstraint, diag(G_plus(extndind,extndind))==G_plus(extndind,nap+1)  ];
        end
        if addsumpConstraint

        end
        typ         = 2;
        cObjective = f_of_xAndAlpha(x_varsdp, alphav,learningparams,typ);
        if addsumpobjective 
%            cObjective = cObjective + learningparams.cp*sum(p);
        end
        if addsumqobjective
%            cObjective = cObjective + learningparams.cq*sum(q);
        end
end
function [x_opt,G_plusv,qv,pv, p_posv, p_negv,beta_pv,eta_pv,misc ] = extractOptResults(cObjective, learningparams, q, G_plus, p, p_pos, p_neg, w_o, g_D_pos, g_D_neg, h, beta_p, eta_p, s1, s2, Options)
    global cnstData
    %% Select Samples to query from : this code not written for part of unlabeled data are being queried from. 
    % Select All of unlabeled samples for Querying
    nap      = cnstData.nap;
    setunlab = cnstData.unlabeled;
    extndind = cnstData.extendInd;
    assert(~isempty(extndind));
    if Options.useq
        qv  = value(q); % q Value may misguide us, because it won't consider y_ui
    else
        qv  = value(G_plus(extndind,nap+1));
    end
    G_plusv = value(G_plus);
    pv      = value(p);
    p_posv  = value(p_pos);
    p_negv  = value(p_neg);    
    w_ov    = value(w_o);
    misc.g_D_pos = value(g_D_pos);
    g_D_neg = value(g_D_neg);
    misc.qinv    = value(G_plus(extndind,nap+1));
    beta_pv      = value(beta_p);
    eta_pv       = value(eta_p);
    misc.hv      = value(h);
    if Options.useoperators
        s1v     = value(s1);
        s2v     = value(s2);
        st      = [s1v;s2v];
    else
        st      = 0;
    end
    [x_opt] = x_conv_abs_h(G_plusv, p_negv, g_D_neg, w_ov,st);
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