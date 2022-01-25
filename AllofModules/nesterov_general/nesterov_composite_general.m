function [X, iter, min_cost] = nesterov_composite_general(grad, proj, x_init, L, opts, f_x)   
% function [X, iter, min_cost] = fista_general(grad,proj, Xinit, L, opts, calc_F)   
% * A Fast Iterative Shrinkage-Thresholding Algorithm for 
% Linear Inverse Problems.
% * Solve the problem: X = arg min_X F(X) = f(X) + lambda*g(X) where:
%   - X: variable, can be a matrix.
%   - f(X): a smooth convex function with continuously differentiable 
%       with Lipschitz continuous gradient `L(f)` (Lipschitz constant of 
%       the gradient of `f`).
%  INPUT:
%       grad   : a function calculating gradient of f(X) given X.
%       proj   : a function calculating pL(x) -- projection
%       Xinit  : a matrix -- initial guess.
%       L      : a scalar the Lipschitz constant of the gradient of f(X).
%       opts   : a struct
%           opts.lambda  : a regularization parameter, can be either a scalar or
%                           a weighted matrix.
%           opts.max_iter: maximum iterations of the algorithm. 
%                           Default 300.
%           opts.tol     : a tolerance, the algorithm will stop if difference 
%                           between two successive X is smaller than this value. 
%                           Default 1e-8.
%           opts.verbose : showing F(X) after each iteration or not. 
%                           Default false. 
%       calc_F: optional, a function calculating value of F at X 
%               via feval(calc_F, X). 
%  OUTPUT:
%      X        : solution
%      iter     : number of run iterations
%      min_cost : the achieved cost
% Modifications:
% 06/17/2016: set default value for opts.pos = false
% -------------------------------------
% Author: Tiep Vu, thv102, 4/6/2016
% (http://www.personal.psu.edu/thv102/)
% -------------------------------------
%     opts = initOpts(opts);
    if ~isfield(opts, 'max_iter')
        opts.max_iter = 500;
    end
    if ~isfield(opts, 'regul')
        opts.regul = 'l1';
    end     
    if ~isfield(opts, 'pos')
        opts.pos = false;
    end
    
    if ~isfield(opts, 'tol')
        opts.tol = 1e-8;
    end
    
    if ~isfield(opts, 'verbose')
        opts.verbose = false;
    end
    if ~isfield(opts, 'mu_x')
        opts.mu_x = 0;
    end
    Linv = 1/L;    
    %lambdaLiv = opts.lambda*Linv;
    mu_x      = opts.mu_x;
    % opts_shrinkage = opts;
    % opts_shrinkage.lambda = lambdaLiv;
    x_old = x_init;
    y_old = x_init;
    x_til = x_init;
    
    t_old = 0;
    f_sum = 0;
    g_avg = 0;
    iter  = 1;
    cost_old = 1e10;
    %% MAIN LOOP
    
    opts_proj = opts;
    %opts_proj.lambda = lambdaLiv;
    while  iter < opts.max_iter
        t_new = t_old + ((1+mu_x*t_old)+sqrt((1+mu_x*t_old)^2+4*L*(1+mu_x*t_old)*t_old))/(2*L);
        x_bar = (t_old*x_til + (t_new-t_old)*x_old)/t_new;
        
        grad_f_xbar = feval(grad, x_bar);
        %f_sum = t_old/t_new*f_sum + (t_new-t_old)/t_new*feval(f_x, x_bar)- (t_new-t_old)/t_new* trace(grad_f_xbar'*x_bar);
        g_avg = t_old/t_new*g_avg + (t_new-t_old)/t_new*grad_f_xbar;
        
        y_old = x_init - t_new*g_avg;
        x_new = feval(proj, y_old, opts_proj);
        x_til = (t_old*x_til + (t_new-t_old)*x_new)/t_new;
        %% check stop criteria
        e(iter) = norm(x_new - x_old)/(1+norm(x_new));
        if e(iter) < opts.tol
            break;
        end
        %% update
        x_old = x_new;
        t_old = t_new;
        %% show progress
        if opts.verbose
            if nargin ~= 0
                cost_new = feval(f_x, x_new);
                if cost_new <= cost_old 
                    stt = 'YES.';
                else 
                    stt = 'NO, check your code.';
                end
                fprintf('iter = %3d, cost = %f, cost decreases? %s\n', ...
                    iter, cost_new, stt);
                cost_old = cost_new;
            else 
                if mod(iter, 5) == 0
                    fprintf('.');
                end
                if mod(iter, 10) == 0 
                   fprintf('%d', iter);
                end     
            end        
        end 
        iter = iter + 1;
    end
    X = x_new;
    if nargout == 3 
        min_cost = feval(f_x, X);
    end 
end 