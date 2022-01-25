function [ wg, w_og , fval] = simple_complex_pso_search(yl, K, K_o, lambda, lambda_o, cp)
    Y   = diag(yl);
    n        = numel(yl);
    KY  = K*Y; zky = zeros(size(KY));
    KY_o= K_o*Y;
    Aineq = [zky,KY_o;zky,-KY_o];
    bineq = [ones(n,1);ones(n,1)];
    LB       = [-1*ones(n,1);-Inf(n,1)];
    UB       = [ones(n,1);Inf(n,1)];

    lamhalf   = lambda/2;
    lam_ohalf = lambda_o/2;

    function f = objective_func(x)
        if strcmp(x,'init')
            f.Aineq = Aineq;
            f.bineq = bineq;
            f.Aeq   = [];
            f.beq   = [];
            f.LB    = [];%LB;
            f.UB    = [];%UB;
            f.nonlcon = [] ;
            f.options.PopulationSize = 500 ;
%            f.options.PopInitRange = [-500; 500] ;
            f.options.ConstrBoundary = 'absorb' ;
        else
            w   = x(1:n)';
            w_o = x(n+1:2*n)';
            temp= K_o*w_o;
            f   = sum((1-KY_o*w_o).*max(1-KY*w,0)) + lamhalf*w'*K*w + lam_ohalf*w_o'*temp + cp*sum(abs(temp));
        end
    end
    problem  = objective_func('init');
    problem.nvars = 2*n;
    problem.options.PlotFcns = {@psoplotbestf,@psoplotswarmsurf} ;
    if isfield(problem.options,'PopulationType') && ...
        ~strcmp(problem.options.PopulationType,'bitstring')
        problem.options.HybridFcn = @fmincon ;
    end
    if isfield(problem.options,'UseParallel') && ...
        strcmpi(problem.options.UseParallel,'always')
        poolopen = false ;
        if isempty(gcp('nocreate'))
            poolobj = parpool ;
            addAttachedFiles(poolobj,[pwd '/testfcns']) ;
        else
            poolopen = true ;
            pctRunOnAll addpath([pwd '/testfcns']) ;
        end
    end
    problem.fitnessfcn = @objective_func;
    [xOpt,fval,exitflag,output,population,scores] = pso(problem);
    %[x,fval] = ga(@objective_func, 2*n,[],[],[],[],LB,UB,@constraint_func);
    wg       = xOpt(1:n);
    w_og     = xOpt(n+1:2*n);
end

