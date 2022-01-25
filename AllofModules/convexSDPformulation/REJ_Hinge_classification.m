function [model] = REJ_Hinge_classification(learningparams, data_train_x, data_train_y, data_train_noisy)
 % This function implements the following paper:
 % Classification with a Reject Option using a Hinge Loss
 % Peter L. Bartlett & Marten H. Wegkamp
 % 
    % options for making constraints
    Options.constraintForm = 5; Options.addsumpConstraint = false; Options.addsumpobjective = true; Options.addsumqobjective = false;
    Options.diagunlabIneq  = false; Options.diagsetQueryIneq = false; 
    Options.useq           = false;
    Options.useoperators   = false;
    Options.useoperators   = false;
    
    [K, ~ ]      = get_two_KernelMatrix(data_train_x, learningparams.KOptions);
    n            = numel(data_train_y);     % size of data 
    Yl           = data_train_y;
    lambda       = learningparams.lambda;
    d_REJ        = learningparams.d_REJ;
%%  Define YALMIP Variables        
    w            = sdpvar(n,1);
    ksi          = sdpvar(n,1);
    gamma_i      = sdpvar(n,1);
%%  Define Problem, Constraints and Objective  
    [solve] = doOptimize(Options);
    if solve.solproblem == 0 
        model.trainx   = data_train_x;  
        model.trainy   = data_train_y;
        model.n        = numel(data_train_y);
        model.w        = value(w);
        model.obj_opt  = solve.cObjectivev;        
        model.name     = 'WSVM_AdaptivelyWeightedSVM';
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
        Yl           = diag(data_train_y');
        cConstraint  = [];
        cConstr1     = (ksi>=0):'ksi_positive';
        cConstraint  = [cConstraint, cConstr1 ];
        cConstraint  = [cConstraint, gamma_i >= -Yl*(K*w)];
        cConstraint  = [cConstraint, ksi>= ones(n,1)-Yl*(K*w)];
        cObjective   = lambda/2*w'*K*w + sum(ksi + (1-2*d_REJ)/d_REJ*gamma_i);  
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