function [alpha_curr,x_curr_beta, dualvars,solstatus] = ProximalStepXOuterAlphaInner(x_G,alpha_0,dualvars0, operators,learningparams,optparams, progress_func,verbose)
    global cnstData

    %% Setting starting values of variables
    %%%%%%%%%%%%%%%%%%%%%For now assume that we have alpha_opt, let see what happens.
    alpha0        = alpha_opt2;%alpha_alpha0;
    x_k           = x_G;
    dualvars_k    = dualvars0;
    dualvars_pre  = dualvars_k;
    v_k_x         = x_G;
    alpha_curr    = alpha0;
    alpha_pre     = alpha_curr;
    x_pre         = x_k;
    %% Retriving learningparameters for using in iterations
    c_p           = learningparams.cp;
    c_a           = learningparams.ca;
    lambda        = learningparams.lambda;
    lambda_o      = learningparams.lambda_o;
    rhox          = learningparams.rhox;
    rhoalpha      = learningparams.rhoalpha;
    BCGD          = true;
    %% Starting values for Nesterov's coeff's
    A_pre         = 0;
    accumGrad_u   = zeros(cnstData.nConic,1);
    accumGrad_beta= zeros(cnstData.n_S,1);
    accumGrad_s   = 0;
    accumf        = 0;
    accumfgrad    = 0;
    accumgrada    = 0;
    %% Exactness and max iteration parameters
    max_iter      = optparams.stmax_iter;
    max_iterADMM  = optparams.stmax_iterADMM;
    [timeinQuad,timeinSDP,timeinSDPproj,t1,t2,t3,timeprof] = createtimevars();
    convergemeasure = zeros(max_iter,3);
    %% Starting loop    
    converged     = false;
    i             = 1;
    max_iter      = 100;
    while ~converged && (i<=max_iter)
        t      = cputime;
        %% concentrate on gradient of function for x. 
        %% Update A_k
        a_k    =   1+optparams.strongcvxmu*A_pre+ sqrt((1+optparams.strongcvxmu*A_pre)^2+4*optparams.L_x*(1+optparams.strongcvxmu*A_pre)*A_pre)/optparams.L_x;
        A_curr =   A_pre + a_k;
        %% Computing v_k : wmstalpha,alph0,accumGrad,accumf,accumgrada,rho,A_curr,lo,up,tol,maxit
        tic;
        [v_k_x, v_k_dualvars] = argminpsi_x(BCGD,x_k, dualvars_k, x_G,accumGrad_u,accumGrad_beta,accumGrad_s,accumfgrad,...
                                            learningparams,A_curr,optparams);
        [Xapprox,p,q,qyu,w_obeta,st]     = getParts(v_k_x);              
        timeinQuad = timeinQuad + toc;
        iter_x = 1;
        %% update x_k 
        %  x_curr = A_pre/A_curr * x_k + a_k / A_curr* v_k_x;    
        %  dualvars_curr = A_pre/A_curr * dualvars_k + a_k / A_curr* v_k_dualvars;
        update_v_k_x_dual();
        [Xapprox,p,q,qyu,w_obeta,st]     = getParts(x_curr);    
        % we must go one gradient step from beta_curr which seems to be y_k
        % in the original algorithm. 
        % in the call of T_L, first we don't use x value and second, although I don't yet know what is the
        % answer, but each iteration of the algorithm is slower. 
        alpha_L     = alpha_opt2;
        [c_u, c_beta, c_s]= x_obj_compgrad_Grad(x_curr, alpha_L, c_a, c_p, lambda, lambda_o);
        x_k         =  T_L(BCGD,x_curr, dualvars_curr, c_u,c_beta,c_s, learningparams, optparams, x_G);  %alphak = T_L(beta_curr, learningparams, optparams, dualvarsPre, x_G);%alphak    = beta_curr; %or
        [Xapprox,p,q,qyu,w_obeta,st]     = getParts(x_k);    
        dualvars_k  = dualvars_curr; %alphak = T_L(beta_curr, learningparams, optparams, dualvarsPre, x_G);%alphak    = beta_curr; %or
        %% Compute x=(u,st,w_obeta), f(x,alphak), grad f(x,alphak) :(lambda,c_a,alphapre,KE,nSDP,n_S,query,unlabeled)
        proxtime    = tic;    
        %%%%%%%%%%%%%%%%%%%%%For now assume that we have alpha_opt, let see what happens.
        %[alpha_new, f_x]  = proxf_Alpha(learningparams, optparams, alpha0, x_k); 
        alpha_new          = alpha_opt2;
        f_x                = 0;
        % dualvars, f_x, gradf_x, perfProfile 
        [c_u, c_beta, c_s]= x_obj_compgrad_Grad(x_k, alpha_new, c_a, c_p, lambda, lambda_o);
        %profiler(1);
        maxobju     = f_x;        
        %% Computing function \psi_k(\alpha))
        accumGrad_u       = accumGrad_u    + a_k* c_u;
        accumGrad_beta    = accumGrad_beta + a_k* c_beta;
        accumGrad_s       = accumGrad_s    + a_k* c_s;
        norm(c_u)
       
        accumf            = accumf         + a_k* f_x;
        
        accumgrada        = accumgrada     + a_k* c_u'   *x_k.u  + a_k* c_beta'*x_k.w_obeta; %+ a_k* c_s'   *x_k.st:since c_s is zeros.
        accumfgrad        = accumf - accumgrada;
        %% hat(x) update: SDP values update
        alpha_curr        = A_pre/A_curr * alpha_curr  + a_k / A_curr* alpha_new;
        
        [Xapprox,p,q,qyu,w_obeta,st]     = getParts(x_curr);
        profiler(2);
        %% updating performance measures
        updateandprintconvmeasure(verbose);
        if convergemeasure(i,1) < 0.01
            max_iterADMM = max_iterADMM +1;
        end
        %% Prepare for the next iterate
        dualvars_pre         = dualvars_k;
        alpha_pre            = alpha_curr;
        x_pre                = x_k;    
        % update Nesterov's Coeff
        A_pre                = A_curr;
        learningparams.rhox  = learningparams.rhox*optparams.mul;
        i                    = i + 1;
    end    
    timereport = makereporttime();
    %% Computing Error for the last inner teration using method of paper, Kolossoski ->goto ActiveOutlierSaddle8 and previous
    function updateandprintconvmeasure(verbose)
        primalobj            = primalobjfunc(learningparams,learningparams.ca,alpha_curr,x_curr,x_G,p,q);
        dualobj              = 0;%dualobjfunc(learningparams.rhox,accumf,accumgrada,accumGrad,A_curr,alpha0,x_curr);
        ptoEnd               = sum(p(14:18))/sum(p(3:18));        
        convergemeasure(i,1) = norm(alpha_curr-alpha_pre)/(1+norm(alpha_pre));
        convergemeasure(i,2) = norm(dualvars_k.y_EC-dualvars_pre.y_EC);%/norm(dualvarsPre.y_EC);
        convergemeasure(i,3) = norm(dualvars_k.y_EV-dualvars_pre.y_EV);%/norm(dualvarsPre.y_EV);
        convergemeasure(i,4) = norm(dualvars_k.y_IC-dualvars_pre.y_IC);%/norm(dualvarsPre.y_IC);
        convergemeasure(i,5) = norm(dualvars_k.y_IV-dualvars_pre.y_IV);%/norm(dualvarsPre.y_IV);
        convergemeasure(i,6) = norm(x_k.u-x_pre.u)+norm(x_k.st-x_pre.st)+norm(x_k.w_obeta-x_pre.w_obeta);
        convergemeasure(i,7) = convergemeasure(i,1)+convergemeasure(i,2)+convergemeasure(i,3)+convergemeasure(i,4)+convergemeasure(i,5)+convergemeasure(i,6); 
        if verbose
            if (mod(i,10)==1)%|| mod(i,10)==2)
                strtitle = sprintf('iter | conv  | SDPMAtrix| alpha |  y_EC  | y_EV  | y_IC  | y_IV  | stdiff| enddif |itSDP|itA|ptoEn|     gap |primal   |dual');
                disp(strtitle);
            end
            str=sprintf('%4.0d |%7.4f|%7.4f   |%7.4f|%7.4f |%7.4f|%7.4f|%7.4f|%7.4f|%7.4f |%3.0d|%4.3f|%9.6f',...
                i,convergemeasure(i,7),convergemeasure(i,1),convergemeasure(i,6),convergemeasure(i,2),convergemeasure(i,3),...
                convergemeasure(i,4),convergemeasure(i,5),...
                iter_x,ptoEnd,primalobj);
            disp(str);
        end
        proxLength= 0;
        [distX, distwo, distst, distalpha, distalpha2, diffXMat] = compDistwithOptimal(x_opt, alpha_opt,alpha_opt2,  x_k,alpha_curr);
        sumdist = distX;
        if (mod(i,10)==1)%|| mod(i,10)==2)
            outputstr  = sprintf('prox step  |   distU   | distwo    |  distst   | sum dist  |distAlpha | distAlpha2');
            disp(outputstr);
        end
        outputstr  = sprintf('%10.7f |%10.7f |%10.7f |%10.7f |%10.7f |%10.7f|%10.7f ',proxLength,distX, distwo, distst,sumdist, distalpha,distalpha2);
        %disp(outputstr);
    end
    function profiler(type)
        if type ==1
            timeinSDP = timeinSDP+toc(proxtime); 
            timeinSDPproj = timeinSDPproj + perfProfile.timeDetailproxf.SDPprojtime;
            t1 = t1  + perfProfile.timeDetailproxf.t1all;
            t2 = t2  + perfProfile.timeDetailproxf.t2all;
            t3 = t3  + perfProfile.timeDetailproxf.t3all;
        elseif type == 2% Attention: what are these lines for?
            timeprof(i,1) = cputime-t;
            timeprof(i,3) = cputime-t;
            if i>1
                timeprof(i,2) = timeprof(i,1)+timeprof(i-1,2);
                t        = cputime;
                timeprof(i,4) = timeprof(i,3)+timeprof(i-1,4);
            end
        end
    end
    function timereport                                             = makereporttime()
            timereport(1,1) = timeinSDP;
            timereport(1,2) = timeinQuad;
            timereport(1,3) = timeinSDPproj;
            timereport(1,4) = t1;
            timereport(1,5) = t2;
            timereport(1,6) = t3;
    end
    function [timeinQuad,timeinSDP,timeinSDPproj,t1,t2,t3,timeprof] = createtimevars()
        timeinQuad    = 0;
        timeinSDP     = 0;
        timeinSDPproj = 0;
        t1            = 0;
        t2            = 0;
        t3            = 0;
        timeprof       = zeros(max_iter,4); 
    end
    function [x_next, dualvars]                   = argminpsi_x(BCGD,x_k, dualvars_k, x_G,...
                                                                                  accumGrad_u,accumGrad_beta,accumGrad_s,accumfgrad,...
                                                                                  learningparams,A_curr,optparams)
            arho      = 1+learningparams.rhox*A_curr;
            mu_reg    = 0;
            if ~BCGD 
                [v_k_x,dualvars] = solvePsi_k3(x_k,operators,dualvars_k,...
                                      x_G,accumGrad_u,accumGrad_beta,accumGrad_s,accumfgrad,learningparams,optparams,arho,mu_reg);
            else
                [v_k_x,dualvars] = solveBCGDDualProblem(x_k,dualvars_k, x_G,accumGrad_u,accumGrad_beta,accumGrad_s,operators,learningparams,optparams,arho,mu_reg);
            end
            x_next         = v_k_x;
    end
    function [x_next,dualvars] = T_L(BCGD,x_curr, dualvars_curr, c_u,c_beta,c_s, learningparams, optparams, x_G)
            accumfgrad  = 0;
            arho        = learningparams.rhox;
            Lx          = norm(cnstData.K);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For now, it must be calculated exactly. 
            mu_reg      = Lx*5;
            if ~BCGD 
                [x_next,dualvars]=solvePsi_k3(x_curr,operators,dualvars_curr,...
                            x_G,c_u,c_beta,c_s,accumfgrad,learningparams,optparams,arho,mu_reg);
            else
                [x_next,dualvars]  = solveBCGDDualProblem(x_curr,dualvars_curr, x_G,c_u,c_beta,c_s,operators,learningparams,optparams,arho,mu_reg);
            end
    end
    function [v_k_x,dualvars]  = solveBCGDDualProblem(x_k,dualvars, x_G,c_u,c_beta,c_s,unscoperators,learningparams,optparams,arho,mu_reg)
            
            gamma_reg      = arho+mu_reg; 
            Ghat.u         = (1/gamma_reg)*(-c_u    + arho*x_G.u      + mu_reg*x_k.u);
            Ghat.w_obeta   = (1/gamma_reg)*(-cnstData.Qinv*c_beta + arho*x_G.w_obeta+ mu_reg*x_k.w_obeta);
            Ghat.st        = (1/gamma_reg)*(-c_s    + arho*x_G.st     + mu_reg*x_k.st);
            
            gscale         = 1;
            operators      = unscoperators;
            %[gscale,operators] = scaleProblem(learningparams,unscoperators,Ghat);
            
            s_I            = [operators.s_IC;operators.s_IV];
            star           = 1; 
            y_ECtil        = dualvars.y_EC;
            y_EVtil        = dualvars.y_EV;
            y_ICtil        = dualvars.y_IC;
            y_IVtil        = dualvars.y_IV;
            Stil           = dualvars.S   ;

            soltype        = 1;
            %% Step 1
            Ay             = operators.A_EC'* y_ECtil + operators.A_IC'* y_ICtil + operators.A_EV'* y_EVtil + operators.A_IV'* y_IVtil;
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            R              = Ay + Stil + Ghat.u;
            [Xapprox,p,q,qyu]     = getu_Parts(R); 
            [Z,v]          = projon_Conestar(cnstData.extendInd,R, x_G.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
            [Xapprox,p,q,qyu]     = getu_Parts(Z);  
            [y_EC,y_EV]    = x_proxLmu_y_E(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, Stil, y_ECtil, y_EVtil, y_ICtil, y_IVtil,...
                                           Ghat, operators);

            [y_IC,y_IV]    = x_proxLmu_y_I(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, Stil, y_EC, y_EV, y_ICtil, y_IVtil,...
                                           Ghat, operators);

            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;                     
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            S              = -(Ay+Z+Ghat.u);
            S              = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
            [Xapprox,p,q,qyu]     = getu_Parts(S); 
 
            [y_IC,y_IV]    = x_proxLmu_y_I(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                           Ghat, operators);
            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;
            [Xapprox,p,q,qyu]     = getu_Parts(Ay); 
            [y_EC,y_EV]    = x_proxLmu_y_E(soltype,optparams.tol4LinearSys,optparams.maxit4LinearSys,...
                                           Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                           Ghat, operators);
            %% 
            Ay             = operators.A_EC'*y_EC + operators.A_IC'*y_IC + operators.A_EV'*y_EV + operators.A_IV'*y_IV;  
            [Xapprox,p,q,qyu]     = getu_Parts(Ay);
            Xp             = proj_oncones(Ghat.u+(Ay+S+Z),cnstData.nSDP,cnstData.n_S,0);   
            [Xapprox,p,q,qyu]     = getu_Parts(Xp); 
            Y              = proj_onP(Ghat.u+(Ay+S),cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
            [Xapprox,p,q,qyu]     = getu_Parts(Y); 
            v_k_x.st       = gscale*min(Ghat.st-[y_IC;y_IV],s_I);
            v_k_x.w_obeta  = gscale*(Ghat.w_obeta + cnstData.Qinv*(operators.B_EV'*y_EV+operators.B_IV'*y_IV));
            %v_k_x.u        = gscale*proj_oncones(Ghat.u+(Ay+Z),cnstData.nSDP,cnstData.n_S,0);   
            v_k_x.u        = gscale*proj_oncones(Ghat.u+(Ay+S+Z),cnstData.nSDP,cnstData.n_S,0); 
            
            dualvars.y_EC  = y_EC;
            dualvars.y_EV  = y_EV;
            dualvars.y_IC  = y_IC;
            dualvars.y_IV  = y_IV;
            dualvars.S     = S;        
    end    
    function [alpha_new, f_x]  = proxf_Alpha(learningparams, optparams, alpha0, x_k)
        global KG;
        global h_of_x;
        global alphapref;
        global rhop;
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        [l_of_x, G_of_x] = x_obj_get_lG_of_x(x_k);
        rhop      = learningparams.rhoalpha;
        alphapref = alpha0;
        KG        = (cnstData.KE.* G_of_x(1:nap,1:nap)) / lambda;
        h_of_x    = [l_of_x;zeros(nap-n_S,1)];
        tol       = 0.001;
        maxit     = 2000;
        [alpha_new, histout, costdata]  = projbfgs(alpha0,@f_lG_x_alpha,cnstData.up,cnstData.lo,tol,maxit);
        f_x       = costdata(end);
    end
    function update_v_k_x_dual()
        x_curr.u           = A_pre/A_curr * x_k.u              + a_k / A_curr* v_k_x.u;
        x_curr.st          = A_pre/A_curr * x_k.st             + a_k / A_curr* v_k_x.st;
        x_curr.w_obeta     = A_pre/A_curr * x_k.w_obeta        + a_k / A_curr* v_k_x.w_obeta;    
        dualvars_curr.y_EC = A_pre/A_curr * dualvars_k.y_EC    + a_k / A_curr* v_k_dualvars.y_EC;
        dualvars_curr.y_EV = A_pre/A_curr * dualvars_k.y_EV    + a_k / A_curr* v_k_dualvars.y_EV;
        dualvars_curr.y_IV = A_pre/A_curr * dualvars_k.y_IV    + a_k / A_curr* v_k_dualvars.y_IV;
        dualvars_curr.y_IC = A_pre/A_curr * dualvars_k.y_IC    + a_k / A_curr* v_k_dualvars.y_IC;
        dualvars_curr.S    = A_pre/A_curr * dualvars_k.S       + a_k / A_curr* v_k_dualvars.S; 
    end
end
