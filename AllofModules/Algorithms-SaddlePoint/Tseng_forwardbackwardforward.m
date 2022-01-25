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
        a_k    =   1+optparams.strongcvxmu*A_pre+ sqrt((1+optparams.strongcvxmu*A_pre)^2+4*optparams.L_x*(1+optparams.strongcvxmu*A_pre)*A_pre)/optparams.L_x;
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
