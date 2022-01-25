function [ALresult,model_set,Y_set]= ActiveOutlierSaddle13(warmStart,x_primalwmst,y_dualwmst,ModelAndData,learningparams)
%This function computes the saddle point for the objective function 
%this is based on the Nesterov's method for Composite functions. First we 
% define a maximal monotone operator by adding Proximal Terms for
% convex and concave variables to the function. and Then define
% max(concave) function as a function of the concave variable. and Use
% Nesterov's method for optimization of this objective. 
% it works good but it is not fast enough. Also, In this it is not obviMous
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
global rho
global rhoml
global mul
global cnstData
verbose      = true;
drawTimePlot = false;
%% Learning Parameters
% Access lambda, lambda_o and rho using learningparams argument 
%% Optimization Parameters
optparams.stmax_iter      = 10;
optparams.stmax_iterADMM  = 5;
optparams.maxouterprox    = 100; 
optparams.gtol            = 0.0001;  % start with a very large value for tolerance 
optparams.alphatol        = optparams.gtol*10^-3;
optparams.tol_ADMM        = 0.0001;
optparams.strongcvxmu    = learningparams.rho;%0.0001; % strongly convex parameter
optparams.thao           = 1;%1.618;% Best values for these parameters. Donot increase mul. , if these values changes in any iteration we must recompute cholesky factors below
optparams.rhoml          = 1/(learningparams.lambda_o*learningparams.rho+1);
optparams.alphamaxit     = 100;
%% Initialize Optimization 
global operators
[operators] = getConstraints3(learningparams);      

[x_G, alpha_alpha0, dualvarsPre] = initVariables(warmStart); 
load('variablebetw','beta_curr');
alpha_alpha0      = beta_curr;
load('datafile.mat','alphav','G_plusv','pv','qv','inpsolverobjective');

optparams.Lipconstalpha = computeLipSchitz(learningparams);

max_iter        = optparams.stmax_iter; 

convergemeasure = zeros(100,3);
timeprofile     = zeros(100,6);
calphathreshold = 0.1;
%% Starting Proximal Optimization loop
outerproxit = 1;
while outerproxit < optparams.maxouterprox 
    %% Proximal Point Iteration
    [x_next, beta_curr, dualvars, timeprofile(outerproxit,:)] ...
                = ProximalStepAlphaX(x_G,alpha_alpha0,dualvarsPre, optparams,learningparams);
    
    if drawTimePlot, drawTimeProfile(timeinSDP,timeinQuad,timeinSDPproj,t1,t2,t3); end    
    %% Update exactness parameters
    reconvalpha = norm(beta_curr-alpha_alpha0)/(1+norm(alpha_alpha0)); 
    if  reconvalpha < calphathreshold % we need more accurate proximal steps
        max_iter       = max_iter + 1;
        calphathreshold= 0.8*calphathreshold;
        optparams.stmax_iterADMM = optparams.stmax_iterADMM + 1;
    end
    proxLength = norm(x_G.u-x_next.u)+norm(x_G.st-x_next.st)+norm(x_G.w_obeta-x_next.w_obeta);
    outputstr  = sprintf('prox step length: %10.7f\n',proxLength);
    disp(outputstr);

    %% Update Proximal Point and Previous Lagrange Values
    dualvarsPre = dualvars;
    x_G         = x_next;
    alpha_alpha0= beta_curr;
    %% next itertation
    outerproxit = outerproxit + 1;
end
%% End of Proximal loop
%% Show diagrams for performance measures
    function L_f = computeLipSchitz(learningparams)
        normK          = norm(cnstData.KE,'fro');
        L_alpha        = sqrt(cnstData.nap)*normK/learningparams.lambda;
        L_G            = cnstData.nap*normK/learningparams.lambda;
        L_f            = 2*L_G+2*L_alpha^2/learningparams.rho;
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
            beta_curr    = rand (cnstData.nap,1);
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
            alpha_alpha0 = beta_curr;
        end
    end
end
function [x_curr,beta_curr,dualvars,timereport] = ProximalStepAlphaX(x_G,alpha_alpha0,dualvarsPre, optparams,learningparams)
    global cnstData
    verbose = true;
    %% Setting Parameters
    mul           = 1;
    %% Setting starting values of variables
    alpha0        = alpha_alpha0;
    beta_curr     = alpha0;
    v_k           = alpha0;
    x_curr        = x_G;
    x_pre         = x_curr;
    beta_pre      = beta_curr;
    %% Starting values for Nesterov's coeff's
    A_pre         = 0;
    accumGrad     = 0;
    accumf        = 0;
    accumgrada    = 0;
    %% Exactness and max iteration parameters
    max_iter      = optparams.stmax_iter;
    max_iterADMM  = optparams.stmax_iterADMM;
    %% Time Profile variables
    timeinQuad    = 0;
    timeinSDP     = 0;
    timeinSDPproj = 0;
    t1            = 0;
    t2            = 0;
    t3            = 0;
    timeprof       = zeros(max_iter,4);
    convergemeasure= zeros(max_iter,3);
    %% Starting loop    
    converged     = false;
    i             = 2;
    while ~converged && (i<=max_iter)
        
        t = cputime;
        %% Update A_k
        a_k    =   1+optparams.strongcvxmu*A_pre+ sqrt((1+optparams.strongcvxmu*A_pre)^2+4*optparams.Lipconstalpha*(1+optparams.strongcvxmu*A_pre)*A_pre)/optparams.Lipconstalpha;
        A_curr =   A_pre + a_k;
        %% update alphak
        alphak =   A_pre/A_curr* beta_curr + a_k/A_curr* v_k;
        %% Compute x=(u,st,w_obeta), f(x,alphak), grad f(x,alphak) :(lambda,c_a,alphapre,KE,nSDP,n_S,query,unlabeled)
        proxtime = tic;        
        [x_new, dualvars, f_alpha,gradf_alpha,perfProfile ] = proxf_X_directADMM(learningparams, optparams, dualvarsPre, x_G, alphak); 
        
        profiler(1);
        maxobju   = f_alpha;
        %% Computing function \psi_k(\alpha))
        accumGrad = accumGrad + a_k* gradf_alpha;
        accumf    = accumf    + a_k* f_alpha;
        accumgrada= accumgrada+ a_k* gradf_alpha'*alphak;
        % Computing v_k       : wmstalpha,alph0,accumGrad,accumf,accumgrada,rho,A_curr,lo,up,tol,maxit
        wmstalpha = beta_curr;
        tic;
        [v_k, iterAlpha,minobjalpha]   = argminpsi(wmstalpha,alpha0,accumGrad,accumf,accumgrada,...
                                                   learningparams.rho,A_curr,cnstData.lo,cnstData.up,optparams.alphatol,optparams.alphamaxit);
        timeinQuad = timeinQuad + toc;
        %% hat(x) update: SDP values update
        x_curr.u      = A_pre/A_curr * x_curr.u         + a_k / A_curr* x_new.u;
        x_curr.st     = A_pre/A_curr * x_curr.st        + a_k / A_curr* x_new.st;
        x_curr.w_obeta= A_pre/A_curr * x_curr.w_obeta   + a_k / A_curr* x_new.w_obeta;
        %% beta_k update: Saddle variable update
        beta_curr  = A_pre/A_curr* beta_curr    + a_k / A_curr* v_k;
        
        % we must go one gradient step from beta_curr which seems to be y_k
        % in the original algorithm. 
        
        %% previous :just for debug and observation
        Xapprox       = reshape(x_curr.u(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
        p             = x_curr.u(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);
        q             = Xapprox(cnstData.extendInd,cnstData.nSDP);
        qyu           = Xapprox(1:cnstData.n_S,cnstData.nSDP);

        profiler(2);
        %% updating performance measures
        updateandprintconvmeasure(verbose);
        if convergemeasure(i,1) < 0.01
            max_iterADMM = max_iterADMM +1;
        end
        %% Prepare for the next iterate
        dualvarsPre          = dualvars;
        x_pre                = x_curr;
        % update Nesterov's Coeff
        A_pre                = A_curr;
        learningparams.rho   = learningparams.rho*mul;
        i       = i + 1;
    end    
    timereport = makereporttime();
    %% Computing Error for the last inner teration using method of paper, Kolossoski ->goto ActiveOutlierSaddle8 and previous
    function updateandprintconvmeasure(verbose)
        primalobj            = primalobjfunc(learningparams,learningparams.ca,beta_curr,x_curr,x_G,p,q);
        dualobj              = dualobjfunc(learningparams.rho,accumf,accumgrada,accumGrad,A_curr,alpha0,beta_curr);
        ptoEnd               = sum(p(14:18))/sum(p(3:18));        
        convergemeasure(i,1) = norm(x_curr.u-x_pre.u)/(1+norm(x_pre.u));
        convergemeasure(i,2) = norm(dualvars.y_EC-dualvarsPre.y_EC);%/norm(dualvarsPre.y_EC);
        convergemeasure(i,3) = norm(dualvars.y_EV-dualvarsPre.y_EV);%/norm(dualvarsPre.y_EV);
        convergemeasure(i,4) = norm(dualvars.y_IC-dualvarsPre.y_IC);%/norm(dualvarsPre.y_IC);
        convergemeasure(i,5) = norm(dualvars.y_IV-dualvarsPre.y_IV);%/norm(dualvarsPre.y_IV);
        convergemeasure(i,6) = norm(beta_curr-beta_pre);
        convergemeasure(i,7) = convergemeasure(i,1)+convergemeasure(i,2)+convergemeasure(i,3)+convergemeasure(i,4)+convergemeasure(i,5)+convergemeasure(i,6); 
        if verbose
            if (mod(i,10)==2)%|| mod(i,10)==2)
                strtitle = sprintf('iter | conv  | SDPMAtrix| alpha |  y_EC  | y_EV  | y_IC  | y_IV  | stdiff| enddif |itSDP|itA|ptoEn|     gap |primal   |dual');
                disp(strtitle);
            end
            str=sprintf('%4.0d |%7.4f|%7.4f   |%7.4f|%7.4f |%7.4f|%7.4f|%7.4f|%7.4f|%7.4f |%3.0d  |%3.0d|%4.3f|%9.6f|%8.5f|%8.5f',...
                i,convergemeasure(i,7),convergemeasure(i,1),convergemeasure(i,6),convergemeasure(i,2),convergemeasure(i,3),...
                convergemeasure(i,4),convergemeasure(i,5),perfProfile.etallstart,perfProfile.etallend,...
                perfProfile.iterSDP,iterAlpha,ptoEnd,primalobj-dualobj,primalobj,dualobj);
            disp(str);
        end
    end
    function profiler(type)
        if type ==1
            timeinSDP = timeinSDP+toc(proxtime); 
            timeinSDPproj = timeinSDPproj + perfProfile.timeDetailproxf.SDPprojtime;
            t1 = t1  + perfProfile.timeDetailproxf.t1all;
            t2 = t2  + perfProfile.timeDetailproxf.t2all;
            t3 = t3  + perfProfile.timeDetailproxf.t3all;
        elseif type == 2
            timeprof(i,1) = cputime-t;
            timeprof(i,2) = timeprof(i,1)+timeprof(i-1,2);
            t        = cputime;
            timeprof(i,3) = cputime-t;
            timeprof(i,4) = timeprof(i,3)+timeprof(i-1,4);
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
end
function [v_k,iterAlpha,psialpha] = argminpsi(wmstalpha,alph0,accumGrad,accumf,accumgrada,rho,A_curr,lo,up,tol,maxit)
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
function [fout,gout]=psi_alpha(alphav)
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
global mul
global operators
global cnstData
[c_k]  = computecoeff(learningparams.lambda,learningparams.ca,alpha_k);
useyalmip   = 0;                                                        
if useyalmip==1
%% computing using the primal-dual test function 
[u_tplus, w_obetavt, s_tplus] = testdualanefficientinexact10(rho,n_l,n_u,n_S,nSDP,batchSize,n_o,n_lbn,lambda,lambda_o,...
                                                            initL,unlabeled,cnstData.extendIndidx,...
                                                            operators.A_EC,operators.b_EC,operators.A_IC,operators.s_IC,operators.A_EV,operators.b_EV,operators.A_IV,operators.s_IV,operators.B_EV,operators.B_IV,y_ECpre,y_ICpre,y_EVpre,y_IVpre,B_E,B_I,...
                                                            x_G.u,x_G.st,x_G.w_obeta,...
                                                            Yl,K,KE,alpha_k,c_k,Q);
 
   X        = u_tplus;
   w_obetav = w_obetavt;
   st       = s_tplus;
   y_EC = y_ECpre;
   y_IC = y_ICpre;
   y_EV = y_EVpre;
   y_IV = y_IVpre;
   etallstart=0;etallend=0;
   S = Spre;
   return;
end

%% fetch sizes of the problem
nec=norm(full(operators.A_EC));
nev=norm(full(operators.A_EV));
nic=norm(full(operators.A_IC));
niv=norm(full(operators.A_IV));
%% setting parameters
threshold  = 1.5;
epsk       = 0.001;
epseig     = 0.01;
max_iter   = optparams.stmax_iterADMM;
tol        = optparams.tol_ADMM;
maxit      = 100; %pcg maxiteration
rhoml      = 1/(learningparams.lambda_o*learningparams.rho+1);
z          = 0;
soltype    = 3;   % compute y_E and y_I using 1: optimization using yalmip , 2: solve Ax=b using pcg ,3:solve Ax=b using cholesky factorization. 
tol        = 10^(-4);
%% setting initial values for improvement measures
iter    = 1;
diff    = zeros(max_iter,1);
etaP    = zeros(max_iter,1);
etaI    = zeros(max_iter,1);
etaIs   = zeros(max_iter,1);
eta_D   = zeros(max_iter,1);
eta_k   = zeros(max_iter,1);
eta_ks  = zeros(max_iter,1);
eta_c1  = zeros(max_iter,1);
eta_gap = zeros(max_iter,1);
etaall  = zeros(max_iter,1);
nonstop = true;
%% Starting values for u, w_obeta and y_EV,y_EC,y_IV,y_IC
s_I = [operators.s_IC;operators.s_IV];
rho = learningparams.rho;

y_ECtil  = dualvarsPre.y_EC ;
y_EVtil  = dualvarsPre.y_EV ;
y_ICtil  = dualvarsPre.y_IC ;
y_IVtil  = dualvarsPre.y_IV ;
Stil     = dualvarsPre.S    ;
Spre     = Stil;
y_ECpre  = y_ECtil;
y_EVpre  = y_EVtil;
y_ICpre  = y_ICtil;
y_IVpre  = y_IVtil;

totalSDPprojtime = 0;
t1all        = 0;
t2all        = 0;
t3all        = 0;
tic;
nextupdate = 5; 
tk  = 1;
k   = 1;
lastiter = false;
while nonstop && iter <max_iter
    %% Step 1
    Ay          = operators.A_EC*y_ECtil + operators.A_IC*y_ICtil + operators.A_EV*y_EVtil + operators.A_IV*y_IVtil;
    t1  = tic;
    R           = Ay + Stil + c_k+learningparams.rho*x_G.u;
    [Z,v]       = projon_Conestar(cnstData.extendInd,R, x_G.st,operators.s_IC,operators.s_IV, y_ICtil,y_IVtil,cnstData.nSDP,cnstData.n_S);
    [y_EC,y_EV] = proxLmu_y_E(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, Stil, y_ECtil, y_EVtil, y_ICtil, y_IVtil,...
                                   x_G,...
                                   c_k,...
                                   operators);
    
    [y_IC,y_IV] = proxLmu_y_I(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, Stil, y_EC, y_EV, y_ICtil, y_IVtil,...
                                   x_G,...
                                   c_k,...
                                   operators);
    Ay          = operators.A_EC*y_EC + operators.A_IC*y_IC + operators.A_EV*y_EV + operators.A_IV*y_IV;                     
    t1all = t1all + toc(t1); 
    S           = -(Ay+Z+c_k+rho*x_G.u);
    star        = 1;
    tsdp = tic;
    S           = proj_oncones(S,cnstData.nSDP,cnstData.n_S,star);   
    totalSDPprojtime = totalSDPprojtime + toc(tsdp);
    t2   = tic;
    [y_IC,y_IV] = proxLmu_y_I(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                   x_G,...
                                   c_k,...
                                   operators);
    [y_EC,y_EV] = proxLmu_y_E(soltype,tol,maxit,rho,learningparams.lambda_o,...
                                   Z ,  v, S, y_EC, y_EV, y_IC, y_IV,...
                                   x_G,...
                                   c_k,...
                                   operators);
    t2all = t2all + toc(t2);                          
    w_obetav = cnstData.KQinv*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*cnstData.Q*x_G.w_obeta);
    %% Step 2: Update S, y_EC,y_EV,y_IC,y_IV,  
    tkplus   = (1+sqrt(1+4*tk^2))/2;
    betak    = (tk-1)/tkplus;
    Stil     = S    + betak*(S-Spre);
    y_ECtil  = y_EC + betak*(y_EC-y_ECpre);
    y_EVtil  = y_EV + betak*(y_EV-y_EVpre);
    y_ICtil  = y_IC + betak*(y_IC-y_ICpre);
    y_IVtil  = y_IV + betak*(y_IV-y_IVpre);
    y_ECpre  = y_EC;
    y_EVpre  = y_EV;
    y_ICpre  = y_IC;
    y_IVpre  = y_IV;
    Spre     = S;
    tk       = tkplus;
    %% Attention, we need to have a restarting mechanism, see Brendan o,Donoghue slides about restart: slide 8
    %% computing accuracy measures of the iteration
    Ay          = operators.A_EC*y_EC + operators.A_IC*y_IC + operators.A_EV*y_EV + operators.A_IV*y_IV;                     
    t3 = tic;
    X           = proj_oncones(x_G.u+1/rho*(Ay+Z+c_k),cnstData.nSDP,cnstData.n_S,0);   
    Xp          = proj_oncones(x_G.u+1/rho*(Ay+S+Z+c_k),cnstData.nSDP,cnstData.n_S,0);   
    
    Y           = proj_onP(x_G.u+1/rho*(Ay+S+c_k),cnstData.nSDP,cnstData.n_S,cnstData.extendInd);
    
    st          = min(x_G.st-[y_IC;y_IV]/rho,s_I);
    [pobj2]=compprimalobjfunc(rho,learningparams.lambda_o,...
                                  X,st,w_obetav,x_G.u ,x_G.st,x_G.w_obeta,...
                                  cnstData.K,c_k,cnstData.Q);
    Dualobjective(iter) = + rho/(2)*norm(operators.A_EC*y_EC+operators.A_IC*y_IC+operators.A_EV*y_EV+operators.A_IV*y_IV+S+Z+c_k/rho+x_G.u)^2 ...
                          -1/(2*rho)*norm(x_G.u,'fro')^2-(operators.b_EC'*y_EC+operators.b_EV'*y_EV) ...
                          +1/2*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*cnstData.Q*x_G.w_obeta)'*cnstData.KQinv*(operators.B_EV*y_EV+operators.B_IV*y_IV+rho*cnstData.Q*x_G.w_obeta)...
                          -learningparams.lambda_o/2*x_G.w_obeta'*cnstData.K*x_G.w_obeta ...
                          +rho/2*norm(-v+[y_IC;y_IV]-x_G.st/rho)^2-v'*s_I-1/(2*rho)*norm(x_G.st)^2;
    primalobjective(iter) = -c_k'*x_G.u-1/(2*rho)*norm(c_k)^2+rho/2*norm(X-x_G.u-c_k/rho)^2+rho/2*norm(st-x_G.st)^2+learningparams.lambda_o/2*w_obetav'*cnstData.K*w_obetav+rho/2*(w_obetav-x_G.w_obeta)'*cnstData.Q*(w_obetav-x_G.w_obeta);
    
    eta1(iter)  = norm([operators.A_EC'*X-operators.b_EC;operators.A_EV'*X-operators.b_EV+operators.B_EV'*w_obetav])/(1+norm([operators.b_EC;operators.b_EV]));
    eta3(iter)  = norm(st+[-operators.A_IC'*X;-operators.A_IV'*X-operators.B_IV'*w_obetav])/(1+norm(st));
    eta2(iter)  = norm(X-Y)/(1+norm(X));
    %etagap is not computed based on my own formulation of the problem. 
    etagap(iter)      = (primalobjective(iter)+Dualobjective(iter))/(1+abs(primalobjective(iter))+abs(Dualobjective(iter)));
    etaaall(iter) = max ( max(eta1(iter),eta2(iter)),eta3(iter));
    t3all = t3all + toc(t3);
%% commented previous Restarting method: To see this restarting method go to ActiveOutlierSaddle7     
%% check exit and get ready to start next iteration 
    gapnonrel = abs(etagap(iter)*(1+abs(primalobjective(iter))+abs(Dualobjective(iter))));
    if etaaall(iter) <= optparams.tol_ADMM %epsk
       nonstop = false;%lastiter=true;%nonstop = false;
    end
    iter = iter + 1;
end
%% Check Constraints: we can check satisfaction of constraints using this function. 
%[ isok , iseqobj ]=checkconstraints(u_tplus,w_obetavt,s_tplus,Kernel,Yl,initL,unlabeled,nSDP,n_S,n_u,n_q,n_o,c_a,batchSize,lambda_o,inpsolverobjective);
etallstart = etaaall(1);
etallend   = etaaall(iter-1);
dur = toc;

x_next.u       = X;
x_next.st      = st;
x_next.w_obeta = w_obetav;

dualvars.y_EC  = y_EC  ;
dualvars.y_EV  = y_EV  ;
dualvars.y_IC  = y_IC  ;
dualvars.y_IV  = y_IV  ;
dualvars.S     = S;

Gq = X(1:cnstData.nSDP*cnstData.nSDP);
Gq = reshape(Gq,cnstData.nSDP,cnstData.nSDP);
q = Gq(cnstData.n_S+1:cnstData.nSDP-1,cnstData.nSDP);
p = X(cnstData.nSDP*cnstData.nSDP+1:cnstData.nSDP*cnstData.nSDP+cnstData.n_S);

[f_alpha,gradf_alpha]       = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_next,x_G,p,q);
timeDetailproxf.SDPprojtime = totalSDPprojtime;
timeDetailproxf.t1all       = t1all;
timeDetailproxf.t2all       = t2all;
timeDetailproxf.t3all       = t3all;

perfProfile.etallstart      = etallstart;
perfProfile.etallend        = etallend;
perfProfile.timeDetailproxf = timeDetailproxf;
perfProfile.iterSDP         = iter;
%% commented: plots: for it goto Previous Version of this file:ActiveOutlierSaddle7
end
function alphtild = proxf_alpha(Xpre,alphapre,Yl,K,n,lambda,rho)
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
function  [s]   = proj_oncones(s,nSDP,n_S,star)
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
function z = proj_sdp(z,n)
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
function [Z,v] = projon_Conestar(query, R, g,s_IC,s_IV, y_IC,y_IV,nSDP,n_S)
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
function [R] = proj_onP(R,nSDP,n_S,query)
%P a set in which query elements of R, is positive. 
RMat    = reshape(R(1:nSDP*nSDP),nSDP,nSDP);
Rpquery = max(RMat(query,nSDP),0);
RMat(query,nSDP ) = Rpquery;
RMat(nSDP ,query) = Rpquery;
R(1:nSDP*nSDP)    = reshape(RMat,nSDP*nSDP,1);
end
function [operators] = getConstraints3(learningparams)
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
for k = cnstData.extendInd
    R = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
    A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
    s_IV(j,1) = 1;
    j = j + 1;
end
%Constraint: q <= 1-\phi(X)^T w_o
%A_IV  = [A_IV,A_IV];
for k = cnstData.extendInd
    R         = sparse([k,cnstData.nSDP],[cnstData.nSDP,k],[0.5,0.5],cnstData.nSDP,cnstData.nSDP);
    A_IV(:,j) = [reshape(R,cnstData.nSDP*cnstData.nSDP,1)',dummy_pag'];
    s_IV(j,1)   = 1;
    j         = j + 1;
end
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

eye_q  = Iind(setdiff(1:cnstData.n_S,cnstData.initL),1:cnstData.n_S);
I_rep  = [sparse(eye_q);-sparse(eye_q);speye(cnstData.n_S);-speye(cnstData.n_S)];
B_IV   = (I_rep*cnstData.K)';
B_I    = [zeros(cnstData.n_S,n_AIC),  B_IV];
B_E    = [zeros(cnstData.n_S,n_AEC),  B_EV];
 
n_AIV  = j-1;
operators.n_AEC= n_AEC;
operators.n_AEV= n_AEV;
operators.n_AIC= n_AIC;
operators.n_AIV= n_AIV;
operators.A_EC = A_EC;
operators.b_EC = b_EC;
operators.A_IC = A_IC;
operators.s_IC = s_IC;
operators.A_EV = A_EV;
operators.b_EV = b_EV;
operators.A_IV = A_IV;
operators.s_IV = s_IV;
operators.B_EV = B_EV;
operators.B_IV = B_IV;
operators.B_E  = B_E;
operators.B_I  = B_I;
%% Computing cholesky factorization for matrices to be inversed in every iteration of SDP inner Problem.
KQinv          = inv(learningparams.lambda_o*cnstData.K+learningparams.rho*cnstData.Q);
operators.AA_E = [ operators.A_EC'*operators.A_EC,operators.A_EC'*operators.A_EV;operators.A_EV'*operators.A_EC,operators.A_EV'*operators.A_EV];
BEKQinv        = operators.B_E'*KQinv;
BEKQinvBE      = BEKQinv*operators.B_E;
RHSE           = operators.AA_E+learningparams.rho*BEKQinvBE;
operators.LcholE         = chol(RHSE);
operators.A_EA_I= [ A_EC'*A_IC,A_EC'*A_IV;A_EV'*A_IC,A_EV'*A_IV];

operators.AA_I           = [ operators.A_IC'*operators.A_IC,operators.A_IC'*operators.A_IV;operators.A_IV'*operators.A_IC,operators.A_IV'*operators.A_IV];
BIKQinv        = operators.B_I'*KQinv;
BIKQinvBI      = BIKQinv*operators.B_I;

RHSI           = operators.AA_I+learningparams.rho*BIKQinvBI+eye(size(operators.AA_I));
operators.LcholI         = chol(RHSI);
%e12    = [ones(2*n_q,1);zeros(2*n_S,1)];
end
function [y_ECd,y_EVd] = proxLmu_y_E(soltype,tol,maxit,rho,lambda_o,...
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
function [y_ICd,y_IVd] = proxLmu_y_I(soltype,tol,maxit,rho,lambda_o,...
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
function alphtild = proxgradientf_alpha(Xpre,g,alphapre,Yl,K,n,n_S,lambda,rho)
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
MC        = (K.* Xpre(1:n,1:n))/lambda;
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
function [fout,gout]=f_alpha(alphav)
global MC;global lambdaf;global alphapref;global rhop;global g_alpha;
%MC = (K.* Xpre(1:n,1:n))/lambdaf;
fout = -alphav'*(1-g_alpha)+alphav'*MC*alphav+1/(2*rhop)*norm(alphav-alphapref)^2;
gout = -1 + MC*alphav+1/rhop*(alphav-alphapref);
end
function px = kk_proj(x,kku,kkl)
ndim=length(x);
px=zeros(ndim,1);
px=min(kku,x); 
px=max(kkl,px);
end
function [c_k]=computecoeff(lambda,c_a,alphapre)
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
function [pobj]=compprimalobjfunc(arho,lambda_o,...
                                  Xr,s,w_obeta,G,g,w_obetapre,...
                                  K,c_k,Q)

pobj  = -c_k'*Xr+arho/2*norm(Xr-G)^2+arho/2*norm(s-g)^2 ...
              +lambda_o/2*w_obeta'*K*w_obeta+arho/2*(w_obeta-w_obetapre)'*Q*(w_obeta-w_obetapre);
end
function [pobj,gradobj]=compprimalobjgradfunc(learningparams,c_k,alpha_k,x_curr,x_G,p,q)
global cnstData

b       = ones(size(p))-p;
b(cnstData.unlabeled) = b(cnstData.unlabeled)-q;
a       = [b;zeros(cnstData.nSDP-cnstData.n_S-1,1)];
pobj    = -c_k'*x_curr.u+learningparams.rho/2*norm(x_curr.u-x_G.u)^2+learningparams.rho/2*norm(x_curr.st-x_G.st)^2 ...
              +learningparams.lambda_o/2*x_curr.w_obeta'*cnstData.K*x_curr.w_obeta...
              +learningparams.rho/2*(x_curr.w_obeta-x_G.w_obeta)'*cnstData.Q*(x_curr.w_obeta-x_G.w_obeta);

uk      = cnstData.Kuvec.*x_curr.u;
UDOTK   = reshape(uk(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP); 
gradobj = 1/learningparams.lambda* UDOTK(1:cnstData.nSDP-1,1:cnstData.nSDP-1)*alpha_k-a;          
end
function [pobj ] = primalobjfunc(learningparams,c_a,alpha_k,x_curr,x_G,p,q)
global cnstData
    [c_k]    = computecoeff(learningparams.lambda,c_a,alpha_k);
    [pobj,~] = compprimalobjgradfunc(learningparams,c_k,alpha_k,x_curr,x_G,p,q);                           
end
function [dobj] = dualobjfunc(rho,accumf,accumgradfalph,accumgrad,A_k,alph0,alpha_knew)
 
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
function [ isok , iseqobj ]= checkconstraints(u,w_obeta,s,Kernel,Yl,initL,unlabeled,nSDP,n_S,n_u,n_q,n_o,c_a,batchSize,lambda_o,inpsolverobjective)
      nLM    = nSDP*nSDP;
      
      query  = n_S+1:n_S+n_q;
      G_plus = reshape(u(1:nLM),nSDP,nSDP);
      p      = u(nLM+1:nLM+n_S);
      q      = G_plus(unlabeled,nSDP);
      rl     = G_plus(initL,nSDP);   % attention: we removed this variable so checking it is trivail
      r      = ones(n_u,1)-p(unlabeled)-q; % attention: we removed this variable so checking it is trivail
      g_D(initL)   = p(initL);          % attention: we removed this variable so checking it is trivail
      g_D(query)   = zeros(n_q,1);      % attention: we removed this variable so checking it is trivail
      g_D(setunlab)= 1-r;               % attention: we removed this variable so checking it is trivail
      
     %cConstraint1 = [beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
                                        %ok
      cConstraint2= sum(q)==batchSize;
      cConstraint3= [0<=q,q<=1];% constraints on q%%implict 
            % constraints on G_plus       ok
      cConstraint4= [G_plus(nap+1,nap+1)==1,G_plus(initL,nap+1)==rl];
            % it is better to substitute p with y_l.*w_o^T\phi(x_i)
      cConstraint5= [diag(G_plus(initL,initL))==1-p(initL),...
                 diag(G_plus(setunlab,setunlab))==r,...
                 diag(G_plus(setQuery,setQuery))==q];

      cConstraint6= [G_plus(setQuery,nap+1)==q,...
                     g_D(initL)==p(initL),g_D(setQuery)==zeros(n_q,1),g_D(setunlab)==1-r];
      cConstraint7= [-p<=Kernel*w_obeta<=p,p<=1,rl==Yl-Kernel(initL,:)*w_obeta];
        % for absolute value of y_u.*(1-pu).*(1-q)
      cConstraint8= [r>=G_plus(unlabeled,nap+1),r>=-G_plus(unlabeled,nap+1)];
      cConstraint9= [r+q+p(unlabeled)==1];
      noiseper=0;%%%%%%%% for test,
            % Form 1: sum(p)<n_o, as a constraint
      cConstraint10=[sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
            %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
            %cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p(1:n))+norm(1-beta_p(n+1:nap)+eta_p(n+1:nap),1)+c*sum(r);
      isok       = cConstraint10&cConstraint9&cConstraint8&cConstraint7&cConstraint6&cConstraint5&cConstraint4&cConstraint3&cConstraint2&cConstraint1;
      
      cObjective = t+lambda_o*w_obeta'*Kernel*w_obeta/2+sum(beta_p)+sum(eta_p(query))+c_a*sum(r);%+norm(1-r,1);
      iseqobj    = cObjective==inpsolverobjective;

end
%% for the following function go to ActiveOutlierSaddle7.m
% function [x,histout,costdata,itc] = gradproj(x0,f,up,low,tol,maxit)
% projection onto active set