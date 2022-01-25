function [model] = LEARNINGWithRejection_CHR(learningparams, data, idx) 
    data_train_x = data.X(:,idx);
    data_train_y = data.Y(idx);
    data_train_noisy  = data.noisy(idx);
    
    % options for making constraints
    Options.constraintForm = 5; Options.addsumpConstraint = false; Options.addsumpobjective = true; Options.addsumqobjective = false;
    Options.diagunlabIneq = false; Options.diagsetQueryIneq = false; 
    Options.useq = false;
    Options.useoperators = false;
    Options.useoperators = false;

    c_LRJ     = learningparams.c_LRJ;
    beta_LRJ  = learningparams.beta_LRJ;
    alpha_LRJ = learningparams.alpha_LRJ;
    
    [K, K_o ] = get_two_KernelMatrix(data_train_x, learningparams.KOptions);
    n         = numel(data_train_y);     % size of data 
    Yl             = data_train_y;
    lambda_o     = learningparams.lambda_o;    
    lambda       = learningparams.lambda;
%%  Define YALMIP Variables        
    u            = sdpvar(n,1);          % For w_o function 
    b_u          = sdpvar(1);
    w            = sdpvar(n,1);
    b_w          = sdpvar(1,1);
    ksi          = sdpvar(n,1);
%%  Define Problem, Constraints and Objective  
    [solve] = doOptimize(Options);
    if solve.solproblem == 0 
        model.trainx   = data_train_x;  
        model.trainy   = data_train_y;
        model.n        = numel(data_train_y);
        model.w        = value(w);
        model.b_w      = value(b_w);
        model.u        = value(u);
        model.b_u      = value(b_u);
        model.obj_opt  = solve.cObjectivev;        
        model.name     = 'LEARNINGWithRejection_CHR';
    else
        model = struct([]);
    end
    function [solve] = doOptimize(Options)
        global cnstDefs
        [cConstraint, cObjective] = constraintType(Options);
        opts = sdpsettings('verbose', cnstDefs.solver_verbose,'solver',cnstDefs.solver);
        sol = optimize(cConstraint,cObjective, opts);
        [primalfeas, dualfeas] = check(cConstraint);
        solve.solproblem = sol.problem;
        if solve.solproblem == 0    
            solve.cObjectivev = value(cObjective);
        else
            solve.cObjectivev = -Inf;
            assert(solproblem~=0,'Problem is not solved in function doOptimize in module ActiveDConvexRelaxOutlier');
        end
    end
    function [cConstraint, cObjective]    = constraintType(options)
        Yl           = data_train_y;
        cConstraint  = [];
        cConstr1     = [(ksi>=0):'ksi_positive'];
        cConstraint  = [cConstraint, cConstr1 ];

        cConstraint  = [cConstraint, (ksi>= c_LRJ*(ones(n,1)-beta_LRJ*(K_o*u+b_u*ones(n,1)))):'ksi_u'];
        cConstraint  = [cConstraint, (ksi>= ones(n,1) + alpha_LRJ/2*((K_o*u+b_u*ones(n,1))-Yl'.*(K*w+b_w))):'h_lessthan1'];

        cObjective   = lambda/2*w'*K*w + lambda_o/2*u'*K_o*u + sum(ksi);  
    end
%     function [saddleobj, alpha_opt]       = compSaddleResult(x_opt, H, eta_pv, beta_pv)
%         global cnstDefs
%         
%         alphas         = sdpvar(nap,1);
%         x_k            = x_opt;
%         H              = (H+H')/2;
%         H              = proj_sdp(H,nSDP);
%         scObjective    = alphas'*x_opt.opt_vars.h ...
%                          -1/(2*learningparams.lambda)*alphas'*(G.*H(1:nap,1:nap))*alphas ...
%                          -sum(x_opt.opt_vars.h);
%         opts = sdpsettings('verbose', cnstDefs.solver_verbose);        
%         scConstraint   = [alphas>=lo ,alphas<=up];
%         ssol           = optimize(scConstraint,-scObjective, opts);
%         if ssol.problem==0
%             alpha_opt  = value(alphas);
%             rhs        = x_opt.opt_vars.h+eta_pv-beta_pv;
%             alphav     = lambda*pinv(K.*H(1:nap,1:nap))*rhs;
%             objsc      = value(scObjective);            
%             saddleobj  = value(scObjective); %f_of_xAndAlpha(x_opt, alpha_opt,learningparams);
%         else
%             saddleobj      = -Inf;
%         end 
%     end
end