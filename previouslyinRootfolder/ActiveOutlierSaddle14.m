function [ALresult,model_set,Y_set]= ActiveOutlierSaddle14(warmStart,x_primalwmst,y_dualwmst,ModelAndData,learningparams)
%This function computes the saddle point for the objective function 
%this is based on the Nesterov's method for Composite functions. First we 
% define a maximal monotone operator by adding Proximal Terms for
% convex and concave variables to the function. and Then define
% max(concave) function as a function of the concave variable. and Use
% Nesterov's method for optimization of this objective. 
% it works good but it is not fast enough. Also, In this it is not obvious
% what is the best iteration number or error in inner iteration for
% computing SDP Proximal. so, it may waste so much time in computing a
% value that it's exactness is not important, especailly in first iterations. 
% For this reason, I'm going to find a criteria that determines inner
% iteration measure for terminating, to get the fastest computation for the
% overall problem. 
%% ToDo: (in no order)
%       optimize quadratic problem using active-set:done, it is done through  a version of projected bfgs
%       Control inner iterations for SDP problem.
%       Check equalness of the solves with that of Out of Box Solver(interiorpoint)
%       Make it neat and clean.
%       Check Variables which are unnecessary
%       Make it ready for recalculating queryies
%       Add constraints for better approximation of w_o,y_u,....
%% global variables 
global cnstData
verbose      = true;
drawTimePlot = false;
%% Learning Parameters
% Access lambda, lambda_o and rho using learningparams argument 
%% Optimization Parameters
optparams.stmax_iter      = 5;
optparams.stmax_iterADMM  = 30;
optparams.maxouterprox    = 100; 
optparams.gtol            = 0.0001;  % start with a very large value for tolerance 
optparams.alphatol        = optparams.gtol*10^-3;
optparams.tol_ADMM        = 0.0001;
optparams.tol4LinearSys   = 10^-7;
optparams.maxit4LinearSys = 100;
optparams.strongcvxmu     = learningparams.rhox;%0.0001; % strongly convex parameter
optparams.thao            = 1;%1.618;% Best values for these parameters. Donot increase mul. , if these values changes in any iteration we must recompute cholesky factors below
optparams.rhoml           = 1/(learningparams.lambda_o*learningparams.rhox+1);
optparams.alphamaxit      = 100;
optparams.mul             = 1;
%% Initialize Optimization 
x_opt.u       = [reshape(ModelAndData.model_setComp.G,cnstData.nSDP*cnstData.nSDP,1);ModelAndData.model_setComp.p];
x_opt.w_obeta = ModelAndData.model_setComp.w_obeta;
x_opt.st      = 0;
alpha_opt     = ModelAndData.model_setComp.alpha;
alpha_opt2    = ModelAndData.model_setComp.alpha;
global operators
[operators] = getConstraints3(learningparams);      

[x0, alpha0, dualvars0] = initVariables(warmStart); 

optparams.Lipconstalpha = computeLipSchitz(learningparams); 
%optparams.Lipconstalpha = optparams.Lipconstalpha /10; % just for testing
TsengFBF        = 0;
x_is_outer      = 1;
max_iter        = optparams.stmax_iter; 

convergemeasure = zeros(100,3);
timeprofile     = zeros(100,6);
calphathreshold = 0.1;
%% Starting Proximal Optimization loop
outerproxit = 1;
while outerproxit < optparams.maxouterprox 
    %% Proximal Point Iteration
    if TsengFBF==0 
        if x_is_outer == 0
            [x_next, alpha_next, dualvars_next, timeprofile(outerproxit,:)] ...
                    = ProximalStepAlphaX(x0,alpha0,dualvars0, optparams,learningparams,true,x_opt,alpha_opt,alpha_opt2);
        else
            [alpha_next,x_next, dualvars_next, timeprofile(outerproxit,:)] ...
                    = ProximalStepXOuterAlphaInner(x0,alpha0, dualvars0, optparams,learningparams,verbose,x_opt,alpha_opt,alpha_opt2);
        end
    else %this didn't work. 
        [x_next, alpha_next, dualvars_next, timeprofile(outerproxit,:)] ...
                    = Tseng_forwardbackwardforward(x0,alpha0,dualvars0, optparams,learningparams,verbose,x_opt,alpha_opt,alpha_opt2);
    end
    if drawTimePlot, drawTimeProfile(timeinSDP,timeinQuad,timeinSDPproj,t1,t2,t3); end    
    %% Update exactness parameters
    reconvalpha = norm(alpha_next-alpha0)/(1+norm(alpha0)); 
    optparams.stmax_iter = optparams.stmax_iter + 1;
    if  reconvalpha < calphathreshold % we need more accurate proximal steps
        max_iter       = max_iter + 1;
        calphathreshold= 0.8*calphathreshold;
        optparams.stmax_iterADMM = optparams.stmax_iterADMM + 1;
    end
    proxLength = norm(x0.u-x_next.u)+norm(x0.st-x_next.st)+norm(x0.w_obeta-x_next.w_obeta);    
    %% Update Proximal Point and Previous Lagrange Values
    dualvars0  = dualvars_next;
    x0         = x_next;
    alpha0     = alpha_next;
    %% next itertation
    outerproxit = outerproxit + 1;
end
%% End of Proximal loop
%% Show diagrams for performance measures
    function L_f = computeLipSchitz(learningparams)
        normK          = norm(cnstData.KE,'fro');
        L_alpha        = sqrt(cnstData.nap)*normK/learningparams.lambda;
        L_G            = cnstData.nap*normK/learningparams.lambda;
        L_f            = 2*L_G+2*L_alpha^2/learningparams.rhox;
    end
    function [x_G, alpha_alpha0, dualvarsPre] = initVariables(warmStart)
        if warmStart
            %% Setting starting point for Proximal Iteration
            x_G             = x_primalwmst.x;
            alpha_alpha0    = x_primalwmst.alpha;
            dualvarsPre     = y_dualwmst;
        else
            dualvarsPre.y_IV =  zeros(operators.n_AIV,1);
            dualvarsPre.y_EC =  zeros(operators.n_AEC,1);
            dualvarsPre.y_EV =  zeros(operators.n_AEV,1);
            dualvarsPre.y_IC =  zeros(operators.n_AIC,1);
            dualvarsPre.S    =  zeros(cnstData.nConic,1);
            n_I              =  operators.n_AIC + operators.n_AIV;
            alpha_0      = rand (cnstData.nap,1);
            w_obetapre   = zeros(cnstData.n_S,1);
            X            = zeros(cnstData.nSDP,cnstData.nSDP);
            p            = zeros(cnstData.n_S,1);
            a            = zeros(cnstData.n_S,1);
            g            = zeros(cnstData.n_S,1);
            u            = [reshape(X,cnstData.nSDP*cnstData.nSDP,1);p];
            vpre         = zeros(n_I,1);
            stpre        = vpre;
            % Spre     = u;% this is zero for now
            upre         = u;% this is zero for now
            x_G.u        = upre;
            x_G.st       = stpre;
            x_G.w_obeta  = w_obetapre;
            alpha_alpha0 = alpha_0;
        end
    end
end
function [alpha_curr,x_curr_beta,dualvars,timereport] = ProximalStepXOuterAlphaInner(x_G,alpha_alpha0,dualvars0, optparams,learningparams,verbose,x_opt,alpha_opt,alpha_opt2)
    global cnstData
    global operators
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
        a_k    =   1+optparams.strongcvxmu*A_pre+ sqrt((1+optparams.strongcvxmu*A_pre)^2+4*optparams.Lipconstalpha*(1+optparams.strongcvxmu*A_pre)*A_pre)/optparams.Lipconstalpha;
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
function [y_ECd,y_EVd]    = x_proxLmu_y_E(soltype,tol,maxit, Z, v, S, y_ECpre, y_EVpre, y_IC, y_IV,...
                                                 Ghat, operators)
global cnstData                               

n_EC  = size(operators.b_EC,1);
n_EV  = size(operators.b_EV,1);
b_ECl = [operators.b_EC;cnstData.Yl(cnstData.initL>0)];
y_Epre= [y_ECpre;y_EVpre];
y_I   = [y_IC;y_IV];
%% which method is the fastest and the best: it seems that atleast for small problems: Cholesky back and forwth solve. method 2 
c_S_Z_ukdrho = S+Z+Ghat.u;
A_Ecszukdrho = [operators.A_EC*c_S_Z_ukdrho;operators.A_EV*c_S_Z_ukdrho];
if soltype == 2 %% if solve using conjugate gradient
    RHSsp = sparse(RHS);
    L     = ichol(RHSsp);
    [y_E,flag,relres,iter] = pcg(RHS,LHS,tol,maxit,L,L',y_Epre);
    assert(flag==0,'pcg didnot converge in computing y_E')
    y_ECd = y_E(1:n_EC);
    y_EVd = y_E(n_EC+1:n_EC+n_EV);
%     y_E = RHS\LHS;
elseif soltype == 3 %% if solve using cholesky factorization
    y_E1  = operators.LcholE'\LHS;
    y_E   = operators.LcholE\y_E1;
    y_ECd = y_E(1:n_EC);
    y_EVd = y_E(n_EC+1:n_EC+n_EV);
elseif soltype == 1 
    const2= 1/2*(norm(Ghat.u)^2+norm(Ghat.st)^2+ Ghat.w_obeta'*cnstData.Q*Ghat.w_obeta);
    y_EC  = sdpvar(n_EC,1);
    y_EV  = sdpvar(n_EV,1);
    Aysdp = operators.A_EC'*y_EC+operators.A_IC'*y_IC+operators.A_EV'*y_EV+operators.A_IV'*y_IV;
    Bysdp = operators.B_EV'*y_EV+operators.B_IV'*y_IV;
    y_I   = [y_IC;y_IV];
    s_I   = [operators.s_IC;operators.s_IV];
    dcObjective = (operators.b_EC'*y_EC+operators.b_EV'*y_EV)- 1/2*norm(Aysdp+S+Z+Ghat.u)^2 ...
                  - 1/2*(Bysdp+cnstData.Q*Ghat.w_obeta)'*cnstData.Qinv*(Bysdp+cnstData.Q*Ghat.w_obeta)...
                  - 1/2*norm(v+y_I-Ghat.st)^2-v'*s_I+const2;
    dcObjective = - dcObjective;
    sol = optimize([],dcObjective);
    if sol.problem==0 
       dobj = value(dcObjective);
       y_ECd  = value(y_EC); 
       y_EVd  = value(y_EV);
       cnstData.Ay = value(Aysdp);
       cnstData.Z  = Z;
       cnstData.Ghatu = Ghat.u;
       cnstData.S = S;
       cnstData.y_EC = y_ECd;
       cnstData.y_EV = y_EVd;
       cnstData.Aysdpval= operators.A_EC'*y_ECd+operators.A_IC'*y_IC+operators.A_EV'*y_EVd+operators.A_IV'*y_IV;
    end
end    
end
function [y_ICd,y_IVd]           = x_proxLmu_y_I(soltype,tol,maxit,Z ,  v, S, y_EC, y_EV, y_ICpre, y_IVpre,...
                                                 Ghat, operators)

global cnstData
Q      = cnstData.Q;

%% which method is the fastest and the best: it seems that atleast for small problems: Cholesky back and forwth solve. method 2 
y_Ipre       = [y_ICpre;y_IVpre];
y_E          = [y_EC   ;y_EV   ];
n_IC         = size(operators.s_IC,1);
n_IV         = size(operators.s_IV,1);
c_S_Z_ukdrho = S+Z+Ghat.u;
A_Icszukdrho = [operators.A_IC*c_S_Z_ukdrho;operators.A_IV*c_S_Z_ukdrho];

if soltype == 2 %% if solve using conjugate gradient
%%    RHSsp = sparse(RHS);
%%    L     = ichol(RHSsp);
    L  = eye(size(RHS));
    [y_Id,flag,relres,iter] = pcg(RHS,LHS,tol,maxit,L,L',y_Ipre);
    assert(flag==0,'pcg didnot converge in computing y_E')
    y_ICd = y_Id(1:n_IC);
    y_IVd = y_Id(n_IC+1:n_IC+n_IV);
%     y_E = RHS\LHS;
elseif soltype == 3 %% if solve using cholesky factorization
    y_I1  = operators.LcholI'\LHS;
    y_I   = operators.LcholI\y_I1;
    y_ICd = y_I(1:n_IC);
    y_IVd = y_I(n_IC+1:n_IC+n_IV);
elseif soltype == 1 
    const2      = 1/2*(norm(Ghat.u)^2+norm(Ghat.st)^2+ Ghat.w_obeta'*cnstData.Q*Ghat.w_obeta);
    y_IC        = sdpvar(n_IC,1);
    y_IV        = sdpvar(n_IV,1);
    Aysdp       = operators.A_EC'*y_EC+operators.A_IC'*y_IC+operators.A_EV'*y_EV+operators.A_IV'*y_IV;
    Bysdp       = operators.B_EV'*y_EV+operators.B_IV'*y_IV;
    s_I         = [operators.s_IC;operators.s_IV];
    y_I         = [y_IC;y_IV];
    dcObjective = (operators.b_EC'*y_EC+operators.b_EV'*y_EV)- 1/2*norm(Aysdp+S+Z+Ghat.u)^2 ...
             - 1/2*(Bysdp+cnstData.Q*Ghat.w_obeta)'*cnstData.Qinv*(Bysdp+cnstData.Q*Ghat.w_obeta)...
             -1/2*norm(v+y_I-Ghat.st)^2-v'*s_I+const2;
    dcObjective = - dcObjective;
    sol = optimize([],dcObjective);
    if sol.problem==0 
       dobj = value(dcObjective);
       y_ICd  = value(y_IC); 
       y_IVd  = value(y_IV);
       cnstData.Ay = value(Aysdp);
       cnstData.Z  = Z;
       cnstData.Ghatu = Ghat.u
       cnstData.S = S;
       cnstData.y_IC = y_ICd;
       cnstData.y_IV = y_IVd;
       cnstData.Aysdpval= operators.A_EC'*y_EC+operators.A_IC'*y_ICd+operators.A_EV'*y_EV+operators.A_IV'*y_IVd;
    end
end
end
function [fout,gout]             = f_lG_x_alpha(alphav)
    global KG;global h_of_x;global alphapref;global rhop;

    fout = -alphav'*h_of_x + 1/2* alphav'*KG*alphav + rhop/(2)*norm(alphav-alphapref)^2;
    gout = -        h_of_x +              KG*alphav + rhop*(alphav-alphapref);
end
function [l_of_x, G_of_x]        = x_obj_get_lG_of_x(x_k)
        global    cnstData;
        nSDP      = cnstData.nSDP;
        nap       = cnstData.nap;
        n_S       = cnstData.n_S;
        nConic    = cnstData.nConic;
        p         = x_k.u(nSDP*nSDP+1:nConic);
        
        G_of_x    = reshape(x_k.u(1:nSDP*nSDP),nSDP,nSDP);
        q         = G_of_x(n_S+1:nSDP-1,nSDP);
        l_of_x    = 1-p;
        l_of_x(cnstData.query) = l_of_x(cnstData.query)-q;
end
function [c_grad        ]        = x_obj_get_Grad(agrad)

end
function [c_u, c_beta, c_s ]    = x_obj_compgrad_Grad(x_k, alpha_new, c_a, c_p, lambda, lambda_o)
global cnstData
        nSDP      = cnstData.nSDP;
        n_S       = cnstData.n_S;
        c_beta    = lambda_o*cnstData.K*x_k.w_obeta;
        c_s       = 0;
        Aqq       = -(alpha_new(cnstData.query)+c_a);
        A_q       = [zeros(n_S,1);Aqq];
        gU        = [-1/(2*lambda)*(cnstData.KE).*(alpha_new*alpha_new'),A_q;...
                     A_q',0];
        gp        = -(alpha_new(1:n_S)+(c_a+c_p)*1);
        c_u       = [reshape(gU,nSDP*nSDP,1);gp]; %% assuming case 2 in the paper. 
end
function [x_curr,beta_curr,dualvars,timereport] = ProximalStepAlphaX(x_G,alpha_alpha0,dualvarsPre, optparams,learningparams,verbose,x_opt,alpha_opt,alpha_opt2)
    global cnstData
    %% Setting starting values of variables
    alpha0        = alpha_alpha0;
    alphak        = alpha0;
    v_k           = alpha0;
    x_curr        = x_G;
    x_pre         = x_curr;
    alphakpre     = alphak;
    %% Starting values for Nesterov's coeff's
    A_pre         = 0;
    accumGrad     = zeros(cnstData.nap,1);
    accumf        = 0;
    accumgrada    = 0;
    %% Exactness and max iteration parameters
    max_iter      = optparams.stmax_iter;
    max_iterADMM  = optparams.stmax_iterADMM;
    
    [timeinQuad,timeinSDP,timeinSDPproj,t1,t2,t3,timeprof] = createtimevars();
    convergemeasure = zeros(max_iter,3);
    %% Starting loop    
    converged     = false;
    i             = 1;
    while ~converged && (i<=max_iter)
        t = cputime;
        %% Update A_k
        a_k    =   1+optparams.strongcvxmu*A_pre+ sqrt((1+optparams.strongcvxmu*A_pre)^2+4*optparams.Lipconstalpha*(1+optparams.strongcvxmu*A_pre)*A_pre)/optparams.Lipconstalpha;
        A_curr =   A_pre + a_k;
        %% Computing v_k       : wmstalpha,alph0,accumGrad,accumf,accumgrada,rho,A_curr,lo,up,tol,maxit
        tic;
        [v_k, iterAlpha, minobjalpha]   = argminpsi(alphak,alpha0,accumGrad,accumf,...
                                                   learningparams.rhoalpha,A_curr,cnstData.lo,cnstData.up,optparams.alphatol,optparams.alphamaxit);
        timeinQuad = timeinQuad + toc;
        %% update alphak
        beta_curr =   A_pre/A_curr* alphak + a_k/A_curr* v_k;
            
        % we must go one gradient step from beta_curr which seems to be y_k
        % in the original algorithm. 
        % in the call of T_L, first we don't use x value and second, although I don't yet know what is the
        % answer, but each iteration of the algorithm is slower. 
        alphak    = beta_curr; %alphak = T_L(beta_curr, learningparams, optparams, dualvarsPre, x_G);%alphak    = beta_curr; %or 
        %% Compute x=(u,st,w_obeta), f(x,alphak), grad f(x,alphak) :(lambda,c_a,alphapre,KE,nSDP,n_S,query,unlabeled)
        proxtime = tic;        
        alphak   = alpha_opt2; %%%%%%%%%%%%%%%%%% This is for test. 
        [x_new, dualvars, f_alpha,gradf_alpha,perfProfile ] = proxf_X_directADMM(learningparams, optparams, dualvarsPre, x_G, alphak); 
        
        profiler(1);
        maxobju   = f_alpha;
        %% Computing function \psi_k(\alpha))
        accumGrad = accumGrad + a_k* gradf_alpha;
        accumf    = accumf    + a_k* f_alpha;
        accumgrada= accumgrada+ a_k* gradf_alpha'*alphak;
        %% hat(x) update: SDP values update
        x_curr.u      = A_pre/A_curr * x_curr.u         + a_k / A_curr* x_new.u;
        x_curr.st     = A_pre/A_curr * x_curr.st        + a_k / A_curr* x_new.st;
        x_curr.w_obeta= A_pre/A_curr * x_curr.w_obeta   + a_k / A_curr* x_new.w_obeta;
        
        [Xapprox,p,q,qyu,w_obeta,st]     = getParts(x_curr);
        profiler(2);
        %% updating performance measures
        updateandprintconvmeasure(verbose);
        if convergemeasure(i,1) < 0.01
            max_iterADMM = max_iterADMM +1;
        end
        %% Prepare for the next iterate
        dualvarsPre          = dualvars;
        x_pre                = x_curr;
        alphakpre            = alphak;    
        % update Nesterov's Coeff
        A_pre                = A_curr;
        learningparams.rhox   = learningparams.rhox*optparams.mul;
        i       = i + 1;
    end    
    timereport = makereporttime();
    %% Computing Error for the last inner teration using method of paper, Kolossoski ->goto ActiveOutlierSaddle8 and previous
    function updateandprintconvmeasure(verbose)
        primalobj            = primalobjfunc(learningparams,learningparams.ca,beta_curr,x_curr,x_G,p,q);
        dualobj              = dualobjfunc(learningparams.rhox,accumf,accumgrada,accumGrad,A_curr,alpha0,beta_curr);
        ptoEnd               = sum(p(14:18))/sum(p(3:18));        
        convergemeasure(i,1) = norm(x_curr.u-x_pre.u)/(1+norm(x_pre.u));
        convergemeasure(i,2) = norm(dualvars.y_EC-dualvarsPre.y_EC);%/norm(dualvarsPre.y_EC);
        convergemeasure(i,3) = norm(dualvars.y_EV-dualvarsPre.y_EV);%/norm(dualvarsPre.y_EV);
        convergemeasure(i,4) = norm(dualvars.y_IC-dualvarsPre.y_IC);%/norm(dualvarsPre.y_IC);
        convergemeasure(i,5) = norm(dualvars.y_IV-dualvarsPre.y_IV);%/norm(dualvarsPre.y_IV);
        convergemeasure(i,6) = norm(alphak-alphakpre);
        convergemeasure(i,7) = convergemeasure(i,1)+convergemeasure(i,2)+convergemeasure(i,3)+convergemeasure(i,4)+convergemeasure(i,5)+convergemeasure(i,6); 
        if verbose
            if (mod(i,10)==1)%|| mod(i,10)==2)
                strtitle = sprintf('iter | conv  | SDPMAtrix| alpha |  y_EC  | y_EV  | y_IC  | y_IV  | stdiff| enddif |itSDP|itA|ptoEn|     gap |primal   |dual');
                disp(strtitle);
            end
            str=sprintf('%4.0d |%7.4f|%7.4f   |%7.4f|%7.4f |%7.4f|%7.4f|%7.4f|%7.4f|%7.4f |%3.0d  |%3.0d|%4.3f|%9.6f|%8.5f|%8.5f',...
                i,convergemeasure(i,7),convergemeasure(i,1),convergemeasure(i,6),convergemeasure(i,2),convergemeasure(i,3),...
                convergemeasure(i,4),convergemeasure(i,5),perfProfile.etallstart,perfProfile.etallend,...
                perfProfile.iterSDP,iterAlpha,ptoEnd,primalobj-dualobj,primalobj,dualobj);
            disp(str);
        end
        proxLength= 0;
        [distX, distwo, distst, distalpha, distalpha2, diffXMat] = compDistwithOptimal(x_opt, alpha_opt,alpha_opt2, x_curr, alphak);
        sumdist = distX;
        if (mod(i,10)==1)%|| mod(i,10)==2)
            outputstr  = sprintf('prox step  |   distU   | distwo    |  distst   | sum dist  |distAlpha | distAlpha2');
            disp(outputstr);
        end
        outputstr  = sprintf('%10.7f |%10.7f |%10.7f |%10.7f |%10.7f |%10.7f|%10.7f ',proxLength,distX, distwo, distst,sumdist, distalpha,distalpha2);
        disp(outputstr);
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
    function timereport = makereporttime()
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
end
function [x_curr,beta_curr,dualvars,timereport] = Tseng_forwardbackwardforward(x_G,alpha_alpha0,dualvarsPre, optparams,learningparams,verbose,x_opt,alpha_opt,alpha_opt2)
    global cnstData
    % This algorithm didn't work for this problem. 
    %% Setting starting values of variables
    alpha0        = alpha_alpha0;
    alphak        = alpha0;
    v_k           = alpha0;
    x_G           = x_opt;%******************************************************Attention: this is wrong, it is just for a test.
    x_curr        = x_G;
    x_pre         = x_curr;
    alphakpre      = alphak;
    %% Starting values for Nesterov's coeff's
    A_pre         = 0;
    accumGrad     = zeros(cnstData.nap,1);
    accumf        = 0;
    accumgrada    = 0;
    %% Exactness and max iteration parameters
    max_iter      = optparams.stmax_iter;
    max_iterADMM  = optparams.stmax_iterADMM;
    max_iter      = 100;%******************************************************Attention
    alphak        = alpha_opt;%******************************************************Attention: this is just for test.
    [timeinQuad,timeinSDP,timeinSDPproj,t1,t2,t3,timeprof] = createtimevars();
    convergemeasure = zeros(max_iter,3);
    iterAlpha       = 1;
    %% Starting loop    
    converged     = false;
    i             = 1;
    while ~converged && (i<=max_iter)
        t = cputime;
        %% Update A_k
        a_k    =   1+optparams.strongcvxmu*A_pre+ sqrt((1+optparams.strongcvxmu*A_pre)^2+4*optparams.Lipconstalpha*(1+optparams.strongcvxmu*A_pre)*A_pre)/optparams.Lipconstalpha;
        A_curr =   A_pre + a_k;
        %% Computing v_k       : wmstalpha,alph0,accumGrad,accumf,accumgrada,rho,A_curr,lo,up,tol,maxit
        tic;
        %[v_k, iterAlpha,minobjalpha]   = argminpsi(alphak,alpha0,accumGrad,accumf,...
        %                                           learningparams.rho,A_curr,cnstData.lo,cnstData.up,optparams.alphatol,optparams.alphamaxit);
        timeinQuad = timeinQuad + toc;
        %% update alphak
        beta_curr =   A_pre/A_curr* alphak + a_k/A_curr* v_k;
        
        
        % we must go one gradient step from beta_curr which seems to be y_k
        % in the original algorithm. 
        % in the call of T_L, first we don't use x value and second, although I don't yet know what is the
        % answer, but each iteration of the algorithm is slower. 
        % alphak    = beta_curr; %alphak = T_L(beta_curr, learningparams, optparams, dualvarsPre, x_G);%alphak    = beta_curr; %or 
        [l_of_x,G_of_x,Xapprox,p,q,qyu,w_obeta,st]     = xParts(x_curr);
        alphakx   = alphak;
        Balpha    = Balpha_gradalpha(alphakx,l_of_x,G_of_x);
        alphaky   = alphakx - (1/learningparams.rho) * Balpha;
        alphaky    = max(min(alphaky,cnstData.up),cnstData.lo);
        %% Compute x=(u,st,w_obeta), f(x,alphak), grad f(x,alphak) :(lambda,c_a,alphapre,KE,nSDP,n_S,query,unlabeled)
        proxtime  = tic;        
        [x_new, dualvars, f_alpha,gradf_alpha,perfProfile ] = proxf_X_directADMM(learningparams, optparams, dualvarsPre, x_G, alphaky); 
        alphakz   = alphaky; 
        %profiler(1);
        maxobju   = f_alpha;
        [l_of_x,G_of_x,Xapprox,p,q,qyu,w_obeta,st]     = xParts(x_new);
        Balpha    = Balpha_gradalpha(alphakz,l_of_x,G_of_x);
        alphakr   = alphakz - (1/learningparams.rho) * Balpha;
        %alphakr    = max(min(alphakr,cnstData.up),cnstData.lo);
        alphak    = max(min(alphakx-alphaky+alphakr,cnstData.up),cnstData.lo); % project step of Tseng Forward_backward_Forward
        %% Computing function \psi_k(\alpha))
        accumGrad = accumGrad + a_k* gradf_alpha;
        accumf    = accumf    + a_k* f_alpha;
        accumgrada= accumgrada+ a_k* gradf_alpha'*alphak;
        %% hat(x) update: SDP values update
        x_curr.u      = A_pre/A_curr * x_curr.u         + a_k / A_curr* x_new.u;
        x_curr.st     = A_pre/A_curr * x_curr.st        + a_k / A_curr* x_new.st;
        x_curr.w_obeta= A_pre/A_curr * x_curr.w_obeta   + a_k / A_curr* x_new.w_obeta;
        
        [l_of_x,G_of_x,Xapprox,p,q,qyu,w_obeta,st]     = xParts(x_curr);
        %profiler(2);
        %% updating performance measures
        updateandprintconvmeasure(verbose);
        if convergemeasure(i,1) < 0.01
            max_iterADMM = max_iterADMM +1;
        end
        %% Prepare for the next iterate
        dualvarsPre          = dualvars;
        x_pre                = x_curr;
        alphakpre            = alphak;    
        % update Nesterov's Coeff
        A_pre                = 0;%A_curr;% for now, don't use Nesterov's Method. 
        learningparams.rho   = learningparams.rho*optparams.mul;
        i       = i + 1;
    end    
    timereport = makereporttime();
    
    function Balpha = Balpha_gradalpha(alphax,l_of_x,G_of_x)
            Balpha = -1/learningparams.lambda* (cnstData.KE.*G_of_x)*alphax+l_of_x;
    end
    function [l_of_x,G_of_x,Xapprox,p,q,qyu,w_obeta,st]     = xParts(x_curr)
        
            %% previous :just for debug and observation
        Xapprox       = reshape(x_curr.u(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
        p             = x_curr.u(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);
        q             = Xapprox(cnstData.extendInd,cnstData.nSDP);
        qa            = zeros(cnstData.n_S,1);
        qa(cnstData.query) = q;
        l_of_x        = zeros(cnstData.nap,1);
        l_of_x(1:cnstData.n_S)= 1-p-qa;
        G_of_x        = Xapprox(1:cnstData.nSDP-1,1:cnstData.nSDP-1);
        qyu           = Xapprox(1:cnstData.n_S,cnstData.nSDP);
        w_obeta       = x_curr.w_obeta;
        st            = x_curr.st;
    end
    function updateandprintconvmeasure(verbose)
        primalobj            = primalobjfunc(learningparams,learningparams.ca,beta_curr,x_curr,x_G,p,q);
        ptoEnd               = sum(p(14:18))/sum(p(3:18));        
        convergemeasure(i,1) = norm(x_curr.u-x_pre.u)/(1+norm(x_pre.u));
        convergemeasure(i,2) = norm(dualvars.y_EC-dualvarsPre.y_EC);%/norm(dualvarsPre.y_EC);
        convergemeasure(i,3) = norm(dualvars.y_EV-dualvarsPre.y_EV);%/norm(dualvarsPre.y_EV);
        convergemeasure(i,4) = norm(dualvars.y_IC-dualvarsPre.y_IC);%/norm(dualvarsPre.y_IC);
        convergemeasure(i,5) = norm(dualvars.y_IV-dualvarsPre.y_IV);%/norm(dualvarsPre.y_IV);
        convergemeasure(i,6) = norm(alphak-alphakpre);
        convergemeasure(i,7) = convergemeasure(i,1)+convergemeasure(i,2)+convergemeasure(i,3)+convergemeasure(i,4)+convergemeasure(i,5)+convergemeasure(i,6); 
        if verbose
            if (mod(i,10)==1)%|| mod(i,10)==2)
                strtitle = sprintf('iter | conv  | SDPMAtrix| alpha |  y_EC  | y_EV  | y_IC  | y_IV  | stdiff| enddif |itSDP|itA|ptoEn|primal ');
                disp(strtitle);
            end
            str=sprintf('%4.0d |%7.4f|%7.4f   |%7.4f|%7.4f |%7.4f|%7.4f|%7.4f|%7.4f|%4.3f|%8.5f',...
                i,convergemeasure(i,7),convergemeasure(i,1),convergemeasure(i,6),convergemeasure(i,2),convergemeasure(i,3),...
                convergemeasure(i,4),convergemeasure(i,5),...
                iterAlpha,ptoEnd,primalobj);
            disp(str);
        end
        proxLength= 0;
        [distX, distwo, distst, distalpha, distalpha2, diffXMat] = compDistwithOptimal(x_opt, alpha_opt,alpha_opt2, x_curr, alphak);
        sumdist = distX;
        if (mod(i,10)==1)%|| mod(i,10)==2)
            outputstr  = sprintf('prox step  |   distU   | distwo    |  distst   | sum dist  |distAlpha | distAlpha2');
            disp(outputstr);
        end
        outputstr  = sprintf('%10.7f |%10.7f |%10.7f |%10.7f ',proxLength,distX, distalpha,distalpha2);
        disp(outputstr);
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
    function timereport = makereporttime()
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
end
function [distX, distwo, distst, distalpha,distalpha2, diffXMat] = compDistwithOptimal(x_opt, alpha_opt,alpha_opt2, x_curr, alpha_curr)
    [Xapproxcurr,pcurr,qcurr,qyucurr,wocurr,stcurr]     = getParts(x_curr);
    [Xapproxopt,popt,qopt,qyuopt,woopt,stopt]           = getParts(x_opt);
    distX                 = norm(Xapproxcurr-Xapproxopt)/norm(Xapproxopt);
    distwo                = norm(wocurr-woopt);
    distst                = norm(stcurr-stopt); % this value is not correct since we always assign stopt=0 
    distalpha             = norm(alpha_opt-alpha_curr);
    distalpha2             = norm(alpha_opt-alpha_curr)/norm(alpha_curr);
    
    diffXMat              = Xapproxcurr-Xapproxopt;
end
function [Xapprox,p,q,qyu,w_obeta,st]     = getParts(x_curr)
    global cnstData
        %% previous :just for debug and observation
    [Xapprox,p,q,qyu]     = getu_Parts(x_curr.u);
    w_obeta       = x_curr.w_obeta;
    st            = x_curr.st;
end
function [Xapprox,p,q,qyu]     = getu_Parts(u)
    global cnstData
        %% previous :just for debug and observation
    Xapprox       = reshape(u(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
    p             = u(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);
    q             = Xapprox(cnstData.extendInd,cnstData.nSDP);
    qyu           = Xapprox(1:cnstData.n_S,cnstData.nSDP);
end
function [v_k,iterAlpha,psialpha]= argminpsi(wmstalpha,alph0,accumGrad,accumf,rho,A_curr,lo,up,tol,maxit)
yalsolok  = 0;
global    proxParam;
global    alphProx;
global    accumGradProx;

accumGradProx = accumGrad;
proxParam     = (1+A_curr*rho);
alphProx      = alph0;

if yalsolok==0
    [x,histout,costdata,iterAlpha] = projbfgs(wmstalpha,@psi_alpha,up,lo,tol,maxit); % This is two order of magnitude faster than projected gradient
    %[x,histout,costdata,iterAlpha] = gradproj(wmstalpha,@psi_alpha,up,lo,tol,maxit);
    v_k = x;
    psialpha = accumf+accumGrad'*v_k;
elseif yalsolok==1 
    n    = size(alph0,1);
    alph = sdpvar(n,1);
    cObjective = accumGrad'*alph+ proxParam /2*norm(alph-alphProx)^2;
    cConstraint= [alph>=lo,alph<=up];
    ops = sdpsettings('verbose',0);
    sol = optimize(cConstraint,cObjective,ops);
    if sol.problem == 0
        v_k = value(alph);
        iterAlpha=2000;
    else
    
    end
end
end
function [fout,gout]             = psi_alpha(alphav)
    global proxParam;
    global alphProx;
    global accumGradProx;
    fout = accumGradProx'*alphav + proxParam/2*norm(alphav-alphProx)^2;
    gout = accumGradProx         + proxParam  *(alphav-alphProx)      ;
end
function [ x_next, dualvars, f_alpha,gradf_alpha,perfProfile] = proxf_X_directADMM(learningparams, optparams, dualvarsPre,x_G,alpha_k)   
%% Based on algorithm ABCD-1 :paper : An efficient inexact ABCD method for least squares semidefinite programming    
%% global variables
global rhoml
global operators
global cnstData
totalSDPprojtime = 0; t1all  = 0; t2all = 0; t3all = 0; iter = 1;
[c_k]       = computecoeff(learningparams.lambda,learningparams.ca,alpha_k);
useyalmip   = 0;                                                        
if useyalmip==1
%% computing using the primal-dual test function 
   [x_next1,dualvars1] = testdualanefficientinexact10(learningparams,operators,dualvarsPre,...
                                                                x_G,alpha_k,c_k);
    [ cSumq, cqInRange,cNapNap, crl,...
     cdiag_initL,cdiag_unlab,cdiag_query,...
     cqExtendInd,cpKernel,crunlab,crqpEquality, ...
     cSumP,crlKw_obeta]     = checkconstraints(x_next1.u,x_next1.w_obeta,x_next1.st, learningparams);
   dualvars = dualvarsPre;
   Gq = x_next1.u(1:cnstData.nSDP*cnstData.nSDP);
   Gq = reshape(Gq,cnstData.nSDP,cnstData.nSDP);
   q  = Gq(cnstData.n_S+1:cnstData.nSDP-1,cnstData.nSDP);
   p  = x_next1.u(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);
   [f_alpha,gradf_alpha]       = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_next1,x_G,p,q);
   etallstart=0;etallend=0;
   perfProfile = setPerfResults();
    x_next   = x_next1;
    dualvars = dualvars1;
    return;
end
%load('next1','x_next1');
%% setting parameters
star       = 1;
max_iter   = optparams.stmax_iterADMM;
tol        = optparams.tol_ADMM;
maxit      = 100; %pcg maxiteration
rhoml      = 1/(learningparams.lambda_o*learningparams.rhox+1);
soltype    = 3;   % compute y_E and y_I using 1: optimization using yalmip , 2: solve Ax=b using pcg ,3:solve Ax=b using cholesky factorization. 
tol        = 10^(-4);
%% setting initial values for improvement measures
iter       = 1;
Dobjective = zeros(max_iter,1);
Pobjective = zeros(max_iter,1);
etaIC      = zeros(max_iter,1);
etaCone    = zeros(max_iter,1);
etaEC      = zeros(max_iter,1);
etagap     = zeros(max_iter,1);
etaall     = zeros(max_iter,1);
nonstop    = true;
%% Starting values for u, w_obeta and y_EV,y_EC,y_IV,y_IC
s_I        = [operators.s_IC;operators.s_IV];
rho        = learningparams.rhox;

[y_ECtil,y_EVtil,y_ICtil,y_IVtil,Stil] = loaddualvars(dualvarsPre);
Spre       = Stil; y_ECpre    = y_ECtil; y_EVpre    = y_EVtil; y_ICpre    = y_ICtil; y_IVpre    = y_IVtil;

tk  = 1;
tstartloop = tic;
while nonstop && iter <max_iter
    t1  = tic;
    %% Step 1
    Ay          = operators.A_EC*y_ECtil + operators.A_IC*y_ICtil + operators.A_EV*y_EVtil + operators.A_IV*y_IVtil;
    R           = Ay + Stil + c_k+rho*x_G.u;
    [Z,v]       = projon_Conestar(cnstData.extendInd,R, x_G.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
    [y_EC,y_EV] = proxLmu_y_E(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, Stil, y_ECtil, y_EVtil, y_ICtil, y_IVtil,...
                                   x_G, c_k, operators);
    
    [y_IC,y_IV] = proxLmu_y_I(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, Stil, y_EC, y_EV, y_ICtil, y_IVtil,...
                                   x_G, c_k, operators);
                               
    Ay          = operators.A_EC*y_EC + operators.A_IC*y_IC + operators.A_EV*y_EV + operators.A_IV*y_IV;                     
    t1all = t1all + toc(t1); 
    S           = -(Ay+Z+c_k+rho*x_G.u);
    tsdp = tic;
    S           = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
    totalSDPprojtime = totalSDPprojtime + toc(tsdp);
    t2   = tic;
    [y_IC,y_IV] = proxLmu_y_I(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                   x_G, c_k, operators);
    [y_EC,y_EV] = proxLmu_y_E(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                   x_G, c_k, operators);
    t2all       = t2all + toc(t2);                          
    w_obetav    = cnstData.KQinv*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*cnstData.Q*x_G.w_obeta);
    %% Step 2: Update S, y_EC,y_EV,y_IC,y_IV,  
    tkplus      = (1+sqrt(1+4*tk^2))/2;
    betak       = (tk-1)/tkplus;
    Stil        = S    + betak*(S-Spre);
    y_ECtil     = y_EC + betak*(y_EC-y_ECpre);
    y_EVtil     = y_EV + betak*(y_EV-y_EVpre);
    y_ICtil     = y_IC + betak*(y_IC-y_ICpre);
    y_IVtil     = y_IV + betak*(y_IV-y_IVpre);
    y_ECpre     = y_EC;
    y_EVpre     = y_EV;
    y_ICpre     = y_IC;
    y_IVpre     = y_IV;
    Spre        = S;
    tk          = tkplus;
    %% Attention, we need to have a restarting mechanism, see Brendan o,Donoghue slides about restart: slide 8
    %% computing accuracy measures of the iteration
    Ay          = operators.A_EC*y_EC + operators.A_IC*y_IC + operators.A_EV*y_EV + operators.A_IV*y_IV;                     
    t3 = tic;
    X           = proj_oncones(x_G.u+1/rho*(Ay+Z+c_k),cnstData.nSDP,cnstData.n_S,0);   
    Xp          = proj_oncones(x_G.u+1/rho*(Ay+S+Z+c_k),cnstData.nSDP,cnstData.n_S,0);   
    
    Y           = proj_onP(x_G.u+1/rho*(Ay+S+c_k),cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
    
    st          = min(x_G.st-[y_IC;y_IV]/rho,s_I);
    
    [pobj2,Dobjective(iter),Pobjective(iter),...
        etaEC(iter),etaIC(iter),etaCone(iter),etagap(iter),etaall(iter) ] = compConvMeasures();
    t3all = t3all + toc(t3);
%% commented previous Restarting method: To see this restarting method go to ActiveOutlierSaddle7     
%% check exit and get ready to start next iteration 
    gapnonrel   = abs(etagap(iter)*(1+abs(Pobjective(iter))+abs(Dobjective(iter))));
    if etaall(iter) <= optparams.tol_ADMM %epsk
       nonstop  = false;%lastiter=true;%nonstop = false;
    end
    iter = iter + 1;
end
%% Check Constraints: we can check satisfaction of constraints using this function. 
%[ isok , iseqobj ]=checkconstraints(u_tplus,w_obetavt,s_tplus,Kernel,Yl,initL,unlabeled,nSDP,n_S,n_u,n_q,n_o,c_a,batchSize,lambda_o,inpsolverobjective);
etallstart = etaall(1);
etallend   = etaall(iter-1);
dur = toc(tstartloop);

setvars(X,st,w_obetav,y_EC,y_EV,y_IC,y_IV,S);%sets x_next and dualvars

% diffu = norm(x_next.u-x_next1.u);
% diffw = norm(x_next.w_obeta-x_next1.w_obeta);
% diffs = norm(x_next.st-x_next1.st);

Gq = X(1:cnstData.nSDP*cnstData.nSDP);
Gq = reshape(Gq,cnstData.nSDP,cnstData.nSDP);
q  = Gq(cnstData.n_S+1:cnstData.nSDP-1,cnstData.nSDP);
p  = X(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);

 [ cSumq, cqInRange,cNapNap, crl,...
     cdiag_initL,cdiag_unlab,cdiag_query,...
     cqExtendInd,cpKernel,crunlab,crqpEquality, ...
     cSumP,crlKw_obeta]     = checkconstraints(X,w_obetav,st, learningparams);
[f_alpha,gradf_alpha]       = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_next,x_G,p,q);
perfProfile = setPerfResults();

%% commented: plots: for it goto Previous Version of this file:ActiveOutlierSaddle7
    function [pobj2,Dobjective,Pobjective,etaEC,etaIC,etaCone,etagap,etaall] = compConvMeasures()
        [pobj2]   = compprimalobjfunc(rho,learningparams.lambda_o,...
                                  X,st,w_obetav,x_G.u ,x_G.st,x_G.w_obeta,...
                                  cnstData.K,c_k,cnstData.Q);
        Dobjective= + rho/(2)*norm(operators.A_EC*y_EC+operators.A_IC*y_IC+operators.A_EV*y_EV+operators.A_IV*y_IV+S+Z+c_k/rho+x_G.u)^2 ...
                              -1/(2*rho)*norm(x_G.u,'fro')^2-(operators.b_EC'*y_EC+operators.b_EV'*y_EV) ...
                              +1/2*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*cnstData.Q*x_G.w_obeta)'*cnstData.KQinv*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*cnstData.Q*x_G.w_obeta)...
                              -learningparams.lambda_o/2*x_G.w_obeta'*cnstData.K*x_G.w_obeta ...
                              +rho/2*norm(-v+[y_IC;y_IV]-x_G.st/rho)^2-v'*s_I-1/(2*rho)*norm(x_G.st)^2;
        Pobjective= -c_k'*x_G.u-1/(2*rho)*norm(c_k)^2+rho/2*norm(X-x_G.u-c_k/rho)^2+rho/2*norm(st-x_G.st)^2+learningparams.lambda_o/2*w_obetav'*cnstData.K*w_obetav+rho/2*(w_obetav-x_G.w_obeta)'*cnstData.Q*(w_obetav-x_G.w_obeta);

        etaEC     = norm([operators.A_EC'*X-operators.b_EC;operators.A_EV'*X-operators.b_EV+operators.B_EV'*w_obetav])/(1+norm([operators.b_EC;operators.b_EV]));
        etaIC     = norm(st+[-operators.A_IC'*X;-operators.A_IV'*X-operators.B_IV'*w_obetav])/(1+norm(st));
        etaCone   = norm(X-Y)/(1+norm(X));
        %etagap is not computed based on my own formulation of the problem. 
        etagap    = (Pobjective+Dobjective)/(1+abs(Pobjective)+abs(Dobjective));
        etaall    = max ( max(etaEC,etaCone),etaIC);
    end
    function setvars(X,st,w_obetav,y_EC,y_EV,y_IC,y_IV,S)
        x_next.u       = X;
        x_next.st      = st;
        x_next.w_obeta = w_obetav;

        dualvars.y_EC  = y_EC  ;
        dualvars.y_EV  = y_EV  ;
        dualvars.y_IC  = y_IC  ;
        dualvars.y_IV  = y_IV  ;
        dualvars.S     = S;
    end
    function [y_ECtil,y_EVtil,y_ICtil,y_IVtil,Stil] = loaddualvars(dualvarsPre)
        y_ECtil  = dualvarsPre.y_EC ;
        y_EVtil  = dualvarsPre.y_EV ;
        y_ICtil  = dualvarsPre.y_IC ;
        y_IVtil  = dualvarsPre.y_IV ;
        Stil     = dualvarsPre.S    ; 
    end
    function [perfProfile] = setPerfResults()
        timeDetailproxf.SDPprojtime = totalSDPprojtime;
        timeDetailproxf.t1all       = t1all;
        timeDetailproxf.t2all       = t2all;
        timeDetailproxf.t3all       = t3all;
        perfProfile.etallstart      = etallstart;
        perfProfile.etallend        = etallend;
        perfProfile.timeDetailproxf = timeDetailproxf;
        perfProfile.iterSDP         = iter;
    end
end
function alphtild                = proxf_alpha(Xpre,alphapre,Yl,K,n,lambda,rho)
alphv       = sdpvar(n,1);
n_l         = numel(Yl);
MC          = K.* Xpre(1:n,1:n);
cConstraint = [alphv>=0,alphv<=1];
cObjective  = -sum(alphv)+ 1/(2*lambda)*alphv'*MC*alphv+1/(2*rho)* norm(alphv-alphapre)^2;
sol         = optimize(cConstraint,cObjective);
if sol.problem ==0 
    alphtild = value(alphv);
else
    assert(true,'Error in proxf_X');
end
end
function [s]                     = proj_oncones(s,nSDP,n_S,star)
G = s(1:nSDP*nSDP);
G = reshape(G,nSDP,nSDP);
G = (G+G')/2;
G = proj_sdp(G,nSDP);
p = s(nSDP*nSDP+1:nSDP*nSDP+n_S);

if star == 0
    p = max(p,0);                    % project on R+
else
    p = max(p,0);
end
s        = [reshape(G,nSDP*nSDP,1);p];
end
function z                       = proj_sdp(z,n)
if n==0
    return;
elseif n==1
    z = max(z,0);
    return;
end

[V,S] = eig(z);
S = diag(S);

idx = find(S>0);
V = V(:,idx);
S = S(idx);
z = V*diag(S)*V';
end
function [Z,v]                   = projon_Conestar(query, R, g,s_IC,s_IV, y_IC,y_IV,nSDP,n_S)
% These two projections are based on the moreau decomposition
% only project some elements of the matrix and not any of the appended
% vector
RMat    = reshape(R(1:nSDP*nSDP),nSDP,nSDP);
Rpquery = max(RMat(query,nSDP),0);
Rp      = RMat;
Rp(query,nSDP ) = Rpquery;
Rp(nSDP ,query) = Rpquery;
ZMat    = Rp - RMat;
Z       = [reshape(ZMat,nSDP*nSDP,1);zeros(n_S,1)];
y_I     = [y_IC;y_IV];
s_I     = [s_IC;s_IV];
vp      = min(g-y_I,s_I); %\mathcal{K}is \forall s\in K, s<=s_I
v       = vp -(g-y_I);  
end
function [R]                     = proj_onP(R,nSDP,n_S,query)
%P a set in which query elements of R, is positive. 
RMat    = reshape(R(1:nSDP*nSDP),nSDP,nSDP);
Rpquery = max(RMat(query,nSDP),0);
RMat(query,nSDP ) = Rpquery;
RMat(nSDP ,query) = Rpquery;
R(1:nSDP*nSDP)    = reshape(RMat,nSDP*nSDP,1);
end
function [operators]             = getConstraints3(learningparams)
global cnstData
%% This function is comes from testdualanefficientinexact9
dummy_pag = zeros(cnstData.n_S,1);
%% Constraints in A_EC(u) = b_EC;
%A_EC      = sparse(nSDP*nSDP+3*n,2*n_l+n_u+2);%3*n_l+n_u+2
% b_EC      = zeros (2*n_l+n_u+2,1);
j         = 1;  
% cConstraint= [cConstraint,sum(q)==batchSize];% constraints on q 
% Constraint :1^T q = bSize
%n_q       = numel(cnstData.extendInd);

q_ind     = [repmat(cnstData.nSDP,cnstData.n_q,1),cnstData.extendInd'];  % this is the indexes of q
R         = sparse([q_ind(:,1)',q_ind(:,2)'],[q_ind(:,2)',q_ind(:,1)'],repmat(0.5,2*cnstData.n_q,1));
A_EC(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag']; 
b_EC(j)   = cnstData.batchSize;
j         = j+1;
% cConstraint= [cConstraint,G_plus(nap+1,nap+1)==1];
% Constraint: G(nSDP,nSDP) = 1
R          = sparse(cnstData.nSDP,cnstData.nSDP,1,cnstData.nSDP,cnstData.nSDP);
A_EC(:,j)  = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag']; 
b_EC(j,1)  = 1;
j          = j+1;
% cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
%                  diag(G_plus(setunlab,setunlab))==r,...
% Constraint: diag(G_{ll,uu})==1-g, since g and a are removed it is
% equivalent to g=p+q , (i.e,diag(G_{ll,uu})==1-(p+q))
% this is equivalent to diag(G_{ll})+ p_l ==1
ap  = zeros(cnstData.n_S,1);
initL = cnstData.initL(cnstData.initL>0)';
for k=initL
    R         = sparse(k,k,1,cnstData.nSDP,cnstData.nSDP);
    ap(k)= 1;
    A_EC(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap']; 
    b_EC(j,1) = 1;
    ap(k)= 0;
    j = j+1;
end
% this is equivalent to diag(G_{uu})+ p_u + q==1
for k = 1:cnstData.n_u
   ku        = cnstData.unlabeled(k);
   kq        = cnstData.extendInd(k);
   R         = sparse([ku,kq,cnstData.nSDP],[ku,cnstData.nSDP,kq],[1,0.5,0.5],cnstData.nSDP,cnstData.nSDP);
   ap(ku)    = 1;
   A_EC(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
   b_EC(j,1) = 1;
   ap(ku)    = 0;
   j = j+1; 
end
% cConstraint=[cConstraint, diag(G_plus(setQuery,setQuery))==q];
% Constraint: diag(G_qq)==q
for k = cnstData.extendInd
   R         = sparse([k,k,cnstData.nSDP],[k,cnstData.nSDP,k],[1,-0.5,-0.5],cnstData.nSDP,cnstData.nSDP);
   A_EC(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',zeros(cnstData.n_S,1)'];
   b_EC(j,1)   = 0;
   j = j+1; 
end
n_AEC    = j-1;

%% Constraints in A_IC(u)<= b_IC
%A_IC = sparse(nSDP*nSDP+3*n,7*n_u+2+n_l);
%b_IC = zeros(7*n_u+2,1);
j = 1;
% cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
% Constraint: 1^T p <=n_o
ap(:)     = 1;
A_IC(:,j) = [zeros(cnstData.nSDP*cnstData.nSDP,1)',ap'];
s_IC(j,1) = cnstData.n_o;
ap(:)     = 0;
j = j+1;
% cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
% Constraint: 1^T p_l <=n_lbn
assert(cnstData.n_lbn<=cnstData.n_o,'n_lbn, number of noisy labeled points must be less than or equal to n_o, number of noisy labeled points');
assert(cnstData.n_lbn>0   ,'n_lbn, is zero and so constraint qualifications are not correct and dual problem is not equivalent to primal');
%% Commented
% ap(initL) = 1;
% A_IC(:,j) = [zeros(nSDP*nSDP,1)',ap',zeros(2*n,1)'];
% s_IC(j,1)   = n_lbn;
% ap(initL) = 0;
% j = j+1;
% constraint: p>=0
% ap     = zeros(n,1);
% p_ind  = unlabeled;
% ag     = zeros(n,1);
% for k=unlabeled
%     ap(k) = -1;
%     A_IC(:,j)    = [zeros(nSDP*nSDP,1)',ap'];
%     ap(k) = 0;
%     s_IC(j,1)  = 0;
%     j = j+1;
% end
% %cConstraint= [cConstraint,0<=q], where is q<=1? is hidden in p+q<=1,p>=0
% % Constraint :q >=0 == -q <=0 
% aa     = zeros(n,1);
% ap     = zeros(n,1);
% p_ind  = unlabeled;
% ag     = zeros(n,1);
% for k=query
%     R = sparse([nSDP,k],[k,nSDP],[-0.5, -0.5],nSDP,nSDP);
%     A_IC(:,j)    = [reshape(R,nSDP*nSDP,1)',zeros(3*n,1)'];
%     s_IC(j,1)      = 0;
%     j = j+1;
% end
%% 
%cConstraint= [cConstraint,0<=q], where is q<=1? q<=1 is not necessary
%because of the constraint p+q <=1, all are positive. so, this constraint
%is deleted. 
% cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
% Constraint: v <= a= 1-p-q :=> v+p+q <=1 
for k = 1:cnstData.n_u
   ku        = cnstData.unlabeled(k);
   kq        = cnstData.extendInd(k);
   R         = sparse([ku,cnstData.nSDP,kq,cnstData.nSDP],[cnstData.nSDP,ku,cnstData.nSDP,kq],[0.5,0.5,0.5,0.5],cnstData.nSDP,cnstData.nSDP);
   ap(ku)    = 1; 
   A_IC(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
   s_IC(j,1) = 1;
   ap(ku)    = 0;
   j = j+1; 
end
%Constraint: -v - a<=0 :=>-v+p+q<=1
for k = 1:cnstData.n_u
   ku        = cnstData.unlabeled(k);
   kq        = cnstData.extendInd(k);
   R         = sparse([ku,cnstData.nSDP,kq,cnstData.nSDP],[cnstData.nSDP,ku,cnstData.nSDP,kq],[-0.5,-0.5,0.5,0.5],cnstData.nSDP,cnstData.nSDP);
   ap(ku)    = 1; 
   A_IC(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',ap'];
   s_IC(j,1) = 1;
   ap(ku)    = 0;
   j = j+1; 
end
% %Constraint: p+q<=1
% for k = 1:n_u
%    ku        = unlabeled(k);
%    kq        = query(k);
%    R         = sparse([kq,nSDP],[nSDP,kq],[0.5,0.5],nSDP,nSDP);
%    ap(ku)    = 1; 
%    A_IC(:,j) = [reshape(R,nSDP*nSDP,1)',ap',zeros(2*n,1)'];
%    s_IC(j,1) = 1;
%    ap(ku)    = 0;
%    j = j+1; 
% end
n_AIC    = j-1;
%% Constraints in A_EV(u) = b_EV + B_EV \beta
% constraint: v_l = y_l-\Phi(X_l)^T w_o
% b_EV = Yl-\Phi(X_l)^T w_o
%A_EV = sparse(nSDP*nSDP+3*n,n_l);
j=1;
for k = initL
   R = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
   A_EV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
   b_EV(j,1)   = cnstData.Yl(k);
   j = j+1;
end
Iind     = speye(cnstData.n_S,cnstData.n_S);
I_l      = Iind(cnstData.initL(cnstData.initL>0),1:cnstData.n_S);
B_EV     = (I_l*cnstData.K)';

n_AEV    = j-1;
%% Constraints in A_IV(u) <= b_IV
%A_IV = sparse(nSDP*nSDP+3*n,2*n_q+2*n_S);
j    = 1;
% Constraint: q <= 1+\phi(X)^T w_o
% for k = cnstData.extendInd
%     R = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
%     A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
%     s_IV(j,1) = 1;
%     j = j + 1;
% end
% %Constraint: q <= 1-\phi(X)^T w_o
% %A_IV  = [A_IV,A_IV];
% for k = cnstData.extendInd
%     R         = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
%     A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
%     s_IV(j,1)   = 1;
%     j         = j + 1;
% end
% Constraint:  \phi(X)^T w_o  <= p
for k   = 1:cnstData.n_S%unlabeled
   ap(k)    = -1;
   A_IV(:,j)= [zeros(cnstData.nSDP*cnstData.nSDP,1)',ap'];
   s_IV(j,1)   = 0;
   ap(k)    = 0;
   j        = j+1;
end
% Constraint:  -\phi(X)^T w_o  - p <= 0 
for k=1:cnstData.n_S%unlabeled
   ap(k)    = -1;
   A_IV(:,j)= [zeros(cnstData.nSDP*cnstData.nSDP,1)',ap'];
   s_IV(j,1)  = 0;
   ap(k)    = 0;
   j        = j+1;
end
n_AIV  = j-1;
eye_q  = Iind(setdiff(1:cnstData.n_S,cnstData.initL),1:cnstData.n_S);
% I_rep  = [sparse(eye_q);-sparse(eye_q);speye(cnstData.n_S);-speye(cnstData.n_S)];
% B_IV   = (I_rep*cnstData.K)';
I_rep  = [speye(cnstData.n_S);-speye(cnstData.n_S)];
B_IV   = (I_rep*cnstData.K)';
B_I    = [zeros(cnstData.n_S,n_AIC),  B_IV];
B_E    = [zeros(cnstData.n_S,n_AEC),  B_EV];
%% Make Sure Constraint Matrices are independent
% check and find independent columns in AB_IV
% AB_IV       = [A_IV;B_IV];
% AB_EV       = [A_EV;B_EV];
% 
% [AB_IVInd,idxindep]=licols(AB_IV,1e-10); %find a subset of independent columns 
% % update 
% A_IV  = AB_IVInd(1:cnstData.nConic,:);
% B_IV  = AB_IVInd(cnstData.nConic+1:end,:);
% s_IV  = s_IV(idxindep);
% n_AIV = size(A_IV,2);
%% 
operators.n_AEC= n_AEC;
operators.n_AEV= n_AEV;
operators.n_AIC= n_AIC;
operators.n_AIV= n_AIV;
operators.A_EC = A_EC';
operators.b_EC = b_EC;
operators.A_IC = A_IC';
operators.s_IC = s_IC;
operators.A_EV = A_EV';
operators.b_EV = b_EV;
operators.A_IV = A_IV';
operators.s_IV = s_IV;
operators.B_EV = B_EV';
operators.B_IV = B_IV';
operators.B_E  = B_E;
operators.B_I  = B_I;
%% Computing cholesky factorization for matrices to be inversed in every iteration of SDP inner Problem.
KQinv          = inv(learningparams.lambda_o*cnstData.K+learningparams.rhox*cnstData.Q);
operators.AA_E = [ operators.A_EC*operators.A_EC',operators.A_EC*operators.A_EV';operators.A_EV*operators.A_EC',operators.A_EV*operators.A_EV'];
BEKQinv        = operators.B_E'*KQinv;
BEKQinvBE      = BEKQinv*operators.B_E;
RHSE           = operators.AA_E+learningparams.rhox*BEKQinvBE;
operators.LcholE         = chol(RHSE);
operators.A_EA_I= [ A_EC'*A_IC,A_EC'*A_IV;A_EV'*A_IC,A_EV'*A_IV];

operators.AA_I           = [ operators.A_IC*operators.A_IC',operators.A_IC*operators.A_IV';operators.A_IV*operators.A_IC',operators.A_IV*operators.A_IV'];
BIKQinv        = operators.B_I'*KQinv;
BIKQinvBI      = BIKQinv*operators.B_I;

% RHSI           = operators.AA_I+learningparams.rhox*BIKQinvBI+eye(size(operators.AA_I));
% operators.LcholI         = chol(RHSI);
%e12    = [ones(2*n_q,1);zeros(2*n_S,1)];
end
function [y_ECd,y_EVd]           = proxLmu_y_E(soltype,tol,maxit,rho,lambda_o,...
                                   Z ,  v, S, y_ECpre, y_EVpre, y_IC, y_IV,...
                                   x_G,...
                                   c_k,...
                                   operators)
global cnstData                               
rhoml = 1/(lambda_o*rho+1);
n_EC  = size(operators.b_EC,1);
n_EV  = size(operators.b_EV,1);
b_ECl = [operators.b_EC;cnstData.Yl(cnstData.initL>0)];
y_Epre= [y_ECpre;y_EVpre];
y_I   = [y_IC;y_IV];
%% which method is the fastest and the best: it seems that atleast for small problems: Cholesky back and forwth solve. method 2 
%AA_E  = [ A_EC'*A_EC,A_EC'*A_EV;A_EV'*A_EC,A_EV'*A_EV];
BEKQinv = operators.B_E'*cnstData.KQinv;
BEKQinvBE = BEKQinv*operators.B_E;


c_S_Z_ukdrho = c_k+S+Z+rho*x_G.u;
A_Ecszukdrho = [operators.A_EC'*c_S_Z_ukdrho;operators.A_EV'*c_S_Z_ukdrho];
BEKQinvBI    = BEKQinv*operators.B_I;

LHS = rho*b_ECl-operators.A_EA_I*y_I-A_Ecszukdrho-rho*BEKQinvBI*y_I-rho^2*BEKQinv*cnstData.Q*x_G.w_obeta;

RHS = operators.AA_E+rho*BEKQinvBE;

%EKB_E = B_E*K*B_E';

%AAKE  = AA_E+rhoml*EKB_E;
%L     = chol(AAKE);
% R_H1  = ([A_EC'*resA_I;(A_EV'*resA_I)])-b_ECl+rho*B_E*w_oPhi + rho*rhoml*B_E*K*B_I'*y_I;
%y_E   = AAKE\R_H1;
%tic;
% y_1   = L'\R_H1;
% y_E2  = L\y_1;
%tdur2  = toc;
%tic;

% [y_E3] = pcg(AAKE,R_H1,[],[],L,L');
% tdur3  = toc;

% y_EC  = y_E(1:numel(b_EC));
% y_EV  = y_E(numel(b_EC)+1:end);
%%
if soltype == 2 %% if solve using conjugate gradient
    RHSsp = sparse(RHS);
    L     = ichol(RHSsp);
    [y_E,flag,relres,iter] = pcg(RHS,LHS,tol,maxit,L,L',y_Epre);
    assert(flag==0,'pcg didnot converge in computing y_E')
    y_ECd = y_E(1:n_EC);
    y_EVd = y_E(n_EC+1:n_EC+n_EV);
%     y_E = RHS\LHS;
elseif soltype == 3 %% if solve using cholesky factorization
    y_E1  = operators.LcholE'\LHS;
    y_E   = operators.LcholE\y_E1;
    y_ECd = y_E(1:n_EC);
    y_EVd = y_E(n_EC+1:n_EC+n_EV);
elseif soltype == 1 
    % Z    = sdpvar(nSDP,nSDP);
    % zp   = sdpvar(n_S,1);
    % za   = sdpvar(n_S,1);
    % zg   = sdpvar(n_S,1);
    y_EC  = sdpvar(n_EC,1);
    y_EV  = sdpvar(n_EV,1);
    y_I  = [y_IC;y_IV];
    s_I  = [s_IC;s_IV];
    % cConstraint= [ Stil>=0,pdu>=0,v>=0,Zpartq>=0];%,v<=arho*(s_I-gprox)+y_I];%,Z(1:n_S,:)==0,Z(:,1:n_S)==0,Z(n_S+1:nSDP,n_S+1:nSDP)>=0];%,Z(nSDP,query)>=0];
    cObjective = -(operators.b_EC'*y_EC+operators.b_EV'*y_EV) ...
                 + 1/(2*rho)*norm(operators.A_EC*y_EC+operators.A_IC*y_IC+operators.A_EV*y_EV+operators.A_IV*y_IV+S+Z+c_k+x_G.u/rho)^2 ...
                 +1/2*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*Q*x_G.w_obeta)'*KQinv*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*Q*x_G.w_obeta)-lambda_o/2*x_G.w_obeta'*K*x_G.w_obeta ...
                 +1/(2*rho)*norm(-v+y_I-x_G.st/rho)^2-v'*s_I-1/(2*rho)*norm(x_G.u,'fro')^2-1/(2*rho)*norm(x_G.st)^2;

    sol = optimize([],cObjective);
    if sol.problem==0 
       dobj = value(cObjective);
       y_ECd1  = value(y_EC); 
       y_EVd1  = value(y_EV);
       y_E1    = [y_ECd1;y_EVd1];
       norm(y_E-y_E1)
    end
end    
end
function [y_ICd,y_IVd]           = proxLmu_y_I(soltype,tol,maxit,rho,lambda_o,...
                                   Z ,  v, S, y_EC, y_EV, y_ICpre, y_IVpre,...
                                   x_G,...
                                   c_k,...
                                   operators)

global cnstData

rhoml = 1/(lambda_o*rho+1);
%% which method is the fastest and the best: it seems that atleast for small problems: Cholesky back and forwth solve. method 2 
y_Ipre = [y_ICpre;y_IVpre];
y_E    = [y_EC   ;y_EV   ];
%% Commented code solving using YALMIP
n_IC = size(operators.s_IC,1);
n_IV = size(operators.s_IV,1);
BIKQinv = operators.B_I'*cnstData.KQinv;
BIKQinvBI = BIKQinv*operators.B_I;
c_S_Z_ukdrho = c_k+S+Z+rho*x_G.u;
A_Icszukdrho = [operators.A_IC'*c_S_Z_ukdrho;operators.A_IV'*c_S_Z_ukdrho];
BIKQinvBE    = BIKQinv*operators.B_E;
LHS = x_G.st/rho+v-operators.A_EA_I'*y_E-A_Icszukdrho-rho*BIKQinvBE*y_E-rho^2*BIKQinv*cnstData.Q*x_G.w_obeta;
RHS = operators.AA_I+rho*BIKQinvBI+eye(size(operators.AA_I));
%%
if soltype == 2 %% if solve using conjugate gradient
%%    RHSsp = sparse(RHS);
%%    L     = ichol(RHSsp);
    L  = eye(size(RHS));
    [y_Id,flag,relres,iter] = pcg(RHS,LHS,tol,maxit,L,L',y_Ipre);
    assert(flag==0,'pcg didnot converge in computing y_E')
    y_ICd = y_Id(1:n_IC);
    y_IVd = y_Id(n_IC+1:n_IC+n_IV);
%     y_E = RHS\LHS;
elseif soltype == 3 %% if solve using cholesky factorization
    y_I1  = operators.LcholI'\LHS;
    y_I   = operators.LcholI\y_I1;
    y_ICd = y_I(1:n_IC);
    y_IVd = y_I(n_IC+1:n_IC+n_IV);
elseif soltype == 1 
    y_IC = sdpvar(n_IC,1);
    y_IV = sdpvar(n_IV,1);

    s_I  = [s_IC;s_IV];
    y_I  = [y_IC;y_IV];
    cObjective =  -(operators.b_EC'*y_EC+operators.b_EV'*y_EV) ...
                 + rho/2*norm(operators.A_EC*y_EC+operators.A_IC*y_IC+operators.A_EV*y_EV+operators.A_IV*y_IV+S+Z+c_k+x_G.u/rho)^2 ...
                 +1/2*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*Q*x_G.w_obeta)'*KQinv*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*Q*x_G.w_obeta)-lambda_o/2*x_G.w_obeta'*K*x_G.w_obeta ...
                 +rho/2*norm(-v+y_I-x_G.st/rho)^2-v'*s_I-1/(2*rho)*norm(x_G.u,'fro')^2-1/(2*rho)*norm(x_G.st)^2;

    sol = optimize([],cObjective);
    if sol.problem==0 
       dobj = value(cObjective);
       y_ICd1  = value(y_IC); 
       y_IVd1  = value(y_IV);
       y_I2    = [y_ICd1;y_IVd1];
       norm(y_Id-y_I2)
    %    norm(y_ICd-y_ICpre)
    %    norm(y_IVd-y_IVpre)
    end
end
end
function alphtild                = proxgradientf_alpha(Xpre,g,alphapre,Yl,K,n,n_S,lambda,rho)
%this function use projected gradient method to optimize for the function 
% f(\alpha) = -1^T
% \alpha+1/(2\lambda)\alpha^T(MC)\alpha+1(2\rho)*norm(\alpha-\alphapre)^2
global MC;
global g_alpha;
global lambdaf;
global alphapref;
global rhop;
rhop      = rho;
alphapref = alphapre;
lambdaf   = lambda;
MC        = (K.*Xpre(1:n,1:n))/lambda;
g_alpha   = g;
u         = ones(n,1);
l         = [zeros(n_S,1);-ones(n-n_S,1)];
tol       = 0.001;
maxit     = 2000;
[x,histout,costdata] = projbfgs(alphapre,@f_alpha,u,l,tol,maxit);
%[x,histout,costdata] = gradproj(alphapre,@f_alpha,u,l,tol,maxit);
alphtild = x;
return;
%this code is based on projected gradient and I think it's problem is
%because of constant step size, Although I didn't test it with correct step
%size which seem that it must be nestrov accelerated method. 
epsalpha = 0.001;
t_k = 1;
alphak = alphapre;
MC          = (K.* Xpre(1:n,1:n))/lambda;
alphakp = alphak-t_k*(-1+MC*alphak+(1/rho)*(alphak-alphapre)); %gradient step
alphakp = min(u,alphakp);                                               %project step for upper bound
alphakp = max(l,alphakp);
it = 1;max_iter=30;
diff = norm(alphakp-alphak)
while diff>epsalpha&& it<max_iter
    alphak  = alphakp;
    alphakp = alphak-t_k*(-1+MC*alphak+(1/rho)*(alphak-alphapre)); %gradient step
    alphakp = min(u,alphakp);                                               %project step for upper bound
    alphakp = max(l,alphakp);   
    diff = norm(alphakp-alphak)
    it=it+1;
end
end
function [fout,gout]             = f_alpha(alphav)
global MC;global lambdaf;global alphapref;global rhop;global g_alpha;
%MC = (K.* Xpre(1:n,1:n))/lambdaf;
fout = -alphav'*(1-g_alpha)+1/2*alphav'*MC*alphav+1/(2*rhop)*norm(alphav-alphapref)^2;
gout = -(1-g_alpha) + MC*alphav+1/rhop*(alphav-alphapref);
%fout = -alphav'*(1-g_alpha)+alphav'*MC*alphav+1/(2*rhop)*norm(alphav-alphapref)^2;
%gout = -1 + MC*alphav+1/rhop*(alphav-alphapref);
end
function px                      = kk_proj(x,kku,kkl)
ndim=length(x);
px=zeros(ndim,1);
px=min(kku,x); 
px=max(kkl,px);
end
function [c_k]                   = computecoeff(lambda,c_a,alphapre)
global cnstData % KE,nSDP,n_S,query,unlabeled
nSDP  = cnstData.nSDP;
n_S   = cnstData.n_S;
%% Computing Coefficeint: c_k
% c_a   =2;
% in the following lines since g is equal to p+q, coefficeints in
% optimization is computed based on that
C_kMS     = [(alphapre*alphapre').*cnstData.KE/(2*lambda),zeros(nSDP-1,1);zeros(1,nSDP-1),0];% coefficients for G
alphatild = alphapre(1:n_S);
n_q       = numel(cnstData.extendInd);
q_ind     = [repmat(nSDP,n_q,1),cnstData.extendInd'];  % this is the indexes of q
cq        = alphatild(cnstData.unlabeled)+c_a*ones(size(cnstData.unlabeled,1),1);
% coefficients for g (which substituted by g=p+q) so, it is considered for q
% and p seperately. alphatild'*(1-g)=alphatild'*1-alphatild'*q-alphatild'*q 
C_kMq     = sparse([q_ind(:,1)',q_ind(:,2)'],[q_ind(:,2)',q_ind(:,1)'],[cq'/2,cq'/2]);
C_kM      = C_kMS + C_kMq;
cp        = alphatild+c_a*ones(n_S,1);
c_k       = [reshape(C_kM,nSDP*nSDP,1);cp];%ca;cg];%first part for C_kM, the other parts for pag: p,a,g

end
function [pobj]                  = compprimalobjfunc(arho,lambda_o,...
                                  Xr,s,w_obeta,G,g,w_obetapre,...
                                  K,c_k,Q)

pobj  = -c_k'*Xr+arho/2*norm(Xr-G)^2+arho/2*norm(s-g)^2 ...
              +lambda_o/2*w_obeta'*K*w_obeta+arho/2*(w_obeta-w_obetapre)'*Q*(w_obeta-w_obetapre);
end
function [pobj,gradobj]          = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_curr,x_G,p,q)
global cnstData

b       = ones(size(p))-p;
b(cnstData.unlabeled) = b(cnstData.unlabeled)-q;
a       = [b;zeros(cnstData.nSDP-cnstData.n_S-1,1)];
pobj    = -c_k'*x_curr.u+learningparams.rhox/2*norm(x_curr.u-x_G.u)^2+learningparams.rhox/2*norm(x_curr.st-x_G.st)^2 ...
              +learningparams.lambda_o/2*x_curr.w_obeta'*cnstData.K*x_curr.w_obeta...
              +learningparams.rhox/2*(x_curr.w_obeta-x_G.w_obeta)'*cnstData.Q*(x_curr.w_obeta-x_G.w_obeta);

uk      = cnstData.Kuvec.*x_curr.u;
UDOTK   = reshape(uk(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP); 
gradobj = 1/learningparams.lambda* UDOTK(1:cnstData.nSDP-1,1:cnstData.nSDP-1)*alpha_k-a;          
end
function [pobj ]                 = primalobjfunc(learningparams,c_a,alpha_k,x_curr,x_G,p,q)
global cnstData
    [c_k]    = computecoeff(learningparams.lambda,c_a,alpha_k);
    [pobj,~] = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_curr,x_G,p,q);                           
end
function [dobj]                  = dualobjfunc(rho,accumf,accumgradfalph,accumgrad,A_k,alph0,alpha_knew)
 
    dobj = accumf - accumgradfalph+...   % a_k f(\alpha_k)-\gradf(alpha_k)'*\alpha_k
           accumgrad'* alpha_knew+...   % (\sum_i \grad(f(\alpha_i))'* \alpha_knew
           A_k*rho/2*norm(alpha_knew-alph0)^2;
    dobj = dobj /A_k;
end
function drawTimeProfile(timeinSDPall,timeinQuad,timeinSDPproj,t1,t2,t3);                                                 
    plot(timeinSDPall);
    hold on;
    plot(time1);
    hold on;
    plot(time2);
    hold on;
    plot(time3);
    hold off;
end
function [ cSumq, cqInRange,cNapNap, crl, cdiag_initL,cdiag_unlab,cdiag_query,cqExtendInd,cpKernel,crunlab,crqpEquality,cSumP,crlKw_obeta]      = checkconstraints(u,w_obeta,s, learningparams)
global cnstData;
      nSDP      = cnstData.nSDP;
      n_S       = cnstData.n_S;
      n_u       = cnstData.n_u;
      n_q       = cnstData.n_q;
      n_o       = cnstData.n_o;
      c_a       = learningparams.ca;
      batchSize = cnstData.batchSize;
      lambda_o  = learningparams.lambda_o;
      unlabeled = cnstData.unlabeled;
      
      initL     = cnstData.initL(cnstData.initL>0);
      Yl        = cnstData.Yl(initL);
      Kernel    = cnstData.K;
      setunlab  = cnstData.unlabeled';
      extendInd = cnstData.extendInd;
      nap       = cnstData.nap;
      
      nLM       = nSDP*nSDP;
      
      query     = n_S+1:n_S+n_q;
      
      G_plus    = reshape(u(1:nLM),nSDP,nSDP);
      p         = u(nLM+1:nLM+n_S,1);
      q         = G_plus(extendInd,nSDP);
      rl        = G_plus(initL,nSDP);   % attention: we removed this variable so checking it is trivail
      r         = ones(n_u,1)-p(unlabeled)-q; % attention: we removed this variable so checking it is trivail
      g_D(initL)   = p(initL);          % attention: we removed this variable so checking it is trivail
      g_D(query)   = zeros(n_q,1);      % attention: we removed this variable so checking it is trivail
      g_D(setunlab)= 1-r;               % attention: we removed this variable so checking it is trivail
      
     %cConstraint1 = [beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
                                        %ok
      cSumq= abs(sum(q)-batchSize);
      cqInRange  = [0<=q,q<=1];% constraints on q%%implict 
      cqInRange  = ~cqInRange; 
            % constraints on G_plus       ok
      cNapNap     = abs(G_plus(nap+1,nap+1)-1);
      crl         = abs(G_plus(initL,nap+1)-rl);
            % it is better to substitute p with y_l.*w_o^T\phi(x_i)
      cdiag_initL = abs(diag(G_plus(initL,initL))-1+p(initL));
      cdiag_unlab = abs(diag(G_plus(setunlab,setunlab))-r);
      cdiag_query = abs(diag(G_plus(extendInd,extendInd))-q);

      cqExtendInd = [abs(G_plus(extendInd,nap+1)-q)];
      
      cpKernel       = [-p<=Kernel*w_obeta;Kernel*w_obeta<=p;
                     p<=1;];
      cpKernel       = ~cpKernel;               
      crlKw_obeta    = abs(rl-Yl+Kernel(initL,:)*w_obeta);
      
      crunlab     = [r>=G_plus(unlabeled,nap+1);
                     r>=-G_plus(unlabeled,nap+1)];
      crunlab     = ~crunlab;
      crqpEquality= [abs(r+q+p(unlabeled)-1)];
      noiseper=0;%%%%%%%% for test,
            % Form 1: sum(p)<n_o, as a constraint
      cSumP       =max(n_o-sum(p),0); %sum(p(initL))<=n_l*lnoiseper/100,
            %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
            %cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p(1:n))+norm(1-beta_p(n+1:nap)+eta_p(n+1:nap),1)+c*sum(r);
end
%% for the following function go to ActiveOutlierSaddle7.m
% function [x,histout,costdata,itc] = gradproj(x0,f,up,low,tol,maxit)
% projection onto active set