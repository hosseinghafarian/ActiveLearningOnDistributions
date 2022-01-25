function [model] = RSVM_RegularizationOfCaseSpecificParameters(learningparams, data, idx) 
    data_train_x = data.X(:,idx);
    data_train_y = data.Y(idx);
    data_train_noisy  = data.noisy(idx);
    
 % This function implements the following paper:
 % Regularization of Case specific parameters for robustness and efficieny,
 % Statistical Science, 2012. 17 citation
    % options for making constraints
    Options.constraintForm = 5; Options.addsumpConstraint = false; Options.addsumpobjective = true; Options.addsumqobjective = false;
    Options.diagunlabIneq  = false; Options.diagsetQueryIneq = false; 
    Options.useq           = false;
    Options.useoperators   = false;
    Options.useoperators   = false;
  
    noiserate    = 0.1;    
    [K, ~ ]    = get_two_KernelMatrix(data_train_x, learningparams.KOptions);
    n            = numel(data_train_y);     % size of data 
    Yl           = data_train_y;
  
    lambda       = learningparams.lambda;
    lambda_CS    = 0.1;%learningparams.lambda_CaseSpecific;
%%  Define YALMIP Variables        
    w            = sdpvar(n,1);
    b_w          = sdpvar(1,1);
    ksi          = sdpvar(n,1);
    nu_i         = sdpvar(n,1);
%%  Define Problem, Constraints and Objective  
    [solve] = doOptimize(Options);
    if solve.solproblem == 0 
        model.trainx   = data_train_x;  
        model.trainy   = data_train_y;
        model.n        = numel(data_train_y);
        model.w        = value(w);
        model.b_w      = value(b_w);
        model.obj_opt  = solve.cObjectivev;        
        model.name     = 'RSVM_RelaxingSupportVectors4CL';
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
        
        cConstraint  = [cConstraint, ksi>= ones(n,1)-Yl*nu_i-Yl*(K*w+b_w*ones(n,1))];
        cConstraint  = [cConstraint, sum(nu_i) <= n*noiserate];
        cConstraint  = [cConstraint, nu_i>=0 ];
        cObjective   = lambda/2*w'*K*w + sum(ksi)+lambda_CS/2*norm(nu_i)^2;  
    end
end