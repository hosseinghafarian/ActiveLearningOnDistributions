function [model] = SMMtrain(learningparams, data, idx , K_SV_SV)
global cnstDefs
    uselibsvm    = true;
    data_train_y = data.Y(idx);
    
%     distu        = unique(data.F); % what are the unique distnums 
    distidx      = data.F_id(idx);     % which unique distnums are for training
    traini       = ismember(data.F, distidx); % select instaces which belongs to distributions in idx
    vecdata_train_x = data.X(:, traini);
    if nargin == 3
        K  = data.K(idx,idx);% Here we must update the kernel matrix if kernel params changed = data.K(idx,idx);
    else
        K  = K_SV_SV;
    end
    n            = numel(data_train_y);     % size of data 
    Yl           = data_train_y;
    lambda       = learningparams.lambda;
    if ~uselibsvm 
        model = optimize_using_yalmip();
    else
        model = solve_using_libsvm();
        
    end
    model.vectrainx   = vecdata_train_x;  
    model.trainy   = data_train_y;
    model.uF_K     = data.F_id(idx);
    model.idxF_K   = idx;
    model.n        = numel(data_train_y);
    
    function model = solve_using_libsvm()
       [cmdstr] = get_libsvm_cmd(learningparams);
%        if ~cnstDefs.solver_verbose 
%            cmdstr = strcat(cmdstr,' -q ');
%        end
       cmdstr     = strcat(cmdstr,' -t 4 ');
       K_indexed   = [(1:n)',K];
       libsvmmodel = svmtrain(data_train_y', K_indexed, cmdstr);
       model.use_libsvm = true;
       model.libsvmmodel = libsvmmodel;
       model.name     = 'SMM_USINGLIBSVM';
    end
    function [model] = optimize_using_yalmip()
%%  Define YALMIP Variables        
    w            = sdpvar(n,1);
    b_w          = sdpvar(1,1);
    ksi          = sdpvar(n,1);
%%  Define Problem, Constraints and Objective  
    [solve] = doOptimize();
    if solve.solproblem == 0 
        
        model.w        = value(w);
        model.b_w      = value(b_w);
        model.obj_opt  = solve.cObjectivev;        
        model.name     = 'SMM';
    else
        model = struct([]);
    end
    function [solve] = doOptimize()
        
        [cConstraint, cObjective] = constraintType();
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
    function [cConstraint, cObjective]    = constraintType()
        Yl           = diag(data_train_y');
        cConstraint  = [];
        cConstr1     = (ksi>=0):'ksi_positive';
        cConstraint  = [cConstraint, cConstr1 ];
        cConstraint  = [cConstraint, ksi>= ones(n,1)-Yl*(K*w+b_w*ones(n,1))];
        cObjective   = lambda/2*w'*K*w + sum(ksi);  
    end
    end
end