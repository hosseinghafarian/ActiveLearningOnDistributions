function [alpha_k, x_k, dualvars_k ] = ProximalStepAlphaX(x_0, alpha_0, dualvars0, operators,learningparams,optparams, progress_func, verbose)
% This algorithm isn't working (yet). 
    global cnstData
    %%
    max_conv_meas = 10^-4;
    max_rel_gap   = 10^-4;
    max_iter      = 100;
    % Setting starting values of variables
    moditer       = 2;
    alphak        = alpha_0;
    x_curr        = x_0;
    x_hat         = x_curr;
    dualvars_hat  = dualvars0;
    x_hatbar      = x_hat;
    alpha_hat     = alphak;
    alpha_hatbar  = alpha_hat;
    dualvars_pre  = dualvars0;
    % setting objective function
    objectivefunc.primal = @primal_regwo_objective;
    objectivefunc.dual   = @dual_regwo_objective;
    %% Starting values for Nesterov's coeff's
    A_pre         = 1;%0 ; Warning: 0 or 1? I think it's correct to be 1 if starting values are not zero. 
    accumGrad     = zeros(cnstData.nap,1);
    accumf        = 0;
    accumgrada    = 0;
    %% Exactness and max iteration parameters
    max_iterADMM  = optparams.stmax_iterADMM;
    convergemeasure = zeros(max_iter,4);
    dist            = zeros(max_iter,4);
    f_val           = zeros(max_iter,2);
    %% Starting loop    
    converged     = false;
    i             = 1;
    while ~converged && (i<=max_iter)
        % Update A_k
        a_k    =   1+optparams.strongcvxmu*A_pre+ sqrt((1+optparams.strongcvxmu*A_pre)^2+4*optparams.L_x*(1+optparams.strongcvxmu*A_pre)*A_pre)/optparams.L_x;
        A_curr =   A_pre + a_k;
        % Computing v_k       : wmstalpha,alph0,accumGrad,accumf,accumgrada,rho,A_curr,lo,up,tol,maxit
        [v_k, iterAlpha, minobjalpha]   = argminpsi(alphak,alpha_0,accumGrad,accumf,...
                                                   learningparams.rhoalpha,A_curr,cnstData.lo,cnstData.up,optparams.alphatol,optparams.alphamaxit);
        
        % update alphak
        beta_curr =   A_pre/A_curr* alphak + a_k/A_curr* v_k;
            
        % we must go one gradient step from beta_curr which seems to be y_k
        % in the original algorithm. 
        % in the call of T_L, first we don't use x value and second, although I don't yet know what is the
        % answer, but each iteration of the algorithm is slower. 
        alphak    = beta_curr; %alphak = T_L(beta_curr, learningparams, optparams, dualvarsPre, x_G);%alphak    = beta_curr; %or 
        % Compute x=(u,st,w_obeta), f(x,alphak), grad f(x,alphak) :(lambda,c_a,alphapre,KE,nSDP,n_S,query,unlabeled)
        [x_next, dualvars_next, f_alpha, gradf_alpha, solstatus] = prox_f_of_x(max_conv_meas, max_rel_gap   , max_iter,...
                                                                objectivefunc, learningparams, optparams, dualvars_pre, x_0, alphak); 
        % Computing function \psi_k(\alpha))
        accumGrad  = accumGrad  + a_k* gradf_alpha;
        accumf     = accumf     + a_k* f_alpha;
        accumgrada = accumgrada + a_k* gradf_alpha'*alphak;
        % hat(x) update: SDP values update
        x_curr.u        = A_pre/A_curr * x_curr.u         + a_k / A_curr* x_next.u;
        x_curr.st       = A_pre/A_curr * x_curr.st        + a_k / A_curr* x_next.st;
        x_curr.w_obeta  = A_pre/A_curr * x_curr.w_obeta   + a_k / A_curr* x_next.w_obeta;
        
        update_ergodic_vars();
        [Xapprox,p,q,qyu,w_obeta,st]     = getParts(x_hatbar);
        
        % updating performance measures
        [convergemeasure(i,:), dist(i,:), f_val(i,:)] ...
            = progress_func(learningparams,i,moditer, verbose, x_hatbar, x_hatbar_pre, dualvars_next, ...
                            dualvars_next, alpha_hatbar, alpha_hatbar_pre, ...
                            x_curr, alphak); 
        if convergemeasure(i,1) < 0.01
            max_iterADMM = max_iterADMM +1;
        end
        %% Prepare for the next iterate
        dualvars_pre         = dualvars_next;
        x_pre                = x_curr;
        alphakpre            = alphak;    
        % update Nesterov's Coeff
        A_pre                = A_curr;
        i       = i + 1;
    end    
    alpha_k    = alpha_hatbar;
    x_k        = x_hatbar;
    dualvars_k = dualvars_hatbar;
    %% Computing Error for the last inner teration using method of paper, Kolossoski ->goto ActiveOutlierSaddle8 and previous
    function update_ergodic_vars()
        x_hat.u          = x_hat.u           + a_k*x_next.u;
        x_hat.w_obeta    = x_hat.w_obeta     + a_k*x_next.w_obeta;
        x_hat.st         = x_hat.st          + a_k*x_next.st;
        dualvars_hat.y_EC= dualvars_hat.y_EC + a_k*dualvars_next.y_EC;   
        dualvars_hat.y_EV= dualvars_hat.y_EV + a_k*dualvars_next.y_EV;
        dualvars_hat.y_IC= dualvars_hat.y_IC + a_k*dualvars_next.y_IC;   
        dualvars_hat.y_IV= dualvars_hat.y_IV + a_k*dualvars_next.y_IV;
        dualvars_hat.S   = dualvars_hat.S    + a_k*dualvars_next.S;
        dualvars_hat.Z   = dualvars_hat.Z    + a_k*dualvars_next.Z;
        dualvars_hat.v   = dualvars_hat.v    + a_k*dualvars_next.v;
        alpha_hat        = alpha_hat         + a_k*alphak;
        
        x_hatbar_pre     = x_hatbar;
        alpha_hatbar_pre = alpha_hatbar;
        x_hatbar.u       = x_hat.u            /A_curr;
        x_hatbar.w_obeta = x_hat.w_obeta      /A_curr;
        x_hatbar.st      = x_hat.st           /A_curr;
        dualvars_hatbar.y_EC= dualvars_hat.y_EC  /A_curr;
        dualvars_hatbar.y_EV= dualvars_hat.y_EV  /A_curr;
        dualvars_hatbar.y_IC= dualvars_hat.y_IC  /A_curr;
        dualvars_hatbar.y_IV= dualvars_hat.y_IV  /A_curr;
        dualvars_hatbar.S   = dualvars_hat.S     /A_curr;
        dualvars_hatbar.Z   = dualvars_hat.Z     /A_curr;
        dualvars_hatbar.v   = dualvars_hat.v     /A_curr;
        alpha_hatbar     = alpha_hat          /A_curr;
    end
end