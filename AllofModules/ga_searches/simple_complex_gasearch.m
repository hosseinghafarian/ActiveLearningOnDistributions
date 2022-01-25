function [ wg, w_og , fval] = simple_complex_gasearch(yl, K, K_o, lambda, lambda_o, cp)
%     Y   = diag(yl);
%     KY  = K*Y;
%     KY_o= K_o*Y;
%     lamhalf   = lambda/2;
%     lam_ohalf = lambda_o/2;
%     n         = numel(yl);
%     function y = objective_func(x)
%         w   = x(1:n)';
%         w_o = x(n+1:2*n)';
%         temp= K_o*w_o;
%         y   = sum((1-KY_o*w_o).*max(1-KY*w,0)) + lamhalf*w'*K*w + lam_ohalf*w_o'*temp + cp*sum(abs(temp));
%     end
%     function [c, ceq] = constraint_func(x)
%         ceq = [];
%         w_o = x(n+1:2*n)';
%         temp= K_o*Y*w_o;
%         c   = [temp-1; -temp-1]';         
%     end
%     LB       = [-1*ones(n,1);-Inf(n,1)];
%     UB       = [ones(n,1);Inf(n,1)];
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
    
    [x,fval] = ga(problem);
    wg       = x(1:n);
    w_og     = x(n+1:2*n);
end

