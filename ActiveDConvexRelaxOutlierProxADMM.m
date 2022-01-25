function [qresult,model_set,Y_set]= ActiveDConvexRelaxOutlierProxADMM(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda)
        % Aassume all of unlabeled data are candidates for active learning query points    
        % n_l :size(initL,2);
        % n_u :n-n_l;
        % n_q :n_u;
        % Only changed constraint inequality to equality 
        % diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))<=q to diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))==q
        lambda_o = lambda/10;
        n_o      = 4;
        % code with WARMSTART 
        n   = n_l+n_u;     % size of data 
        c   = 1;           % parameter for finding absolute value of y_u.*(1-q) in cost function 
%% Commented code for now 
%         global Warmstart;
%         global WMST_beforeStart ;
%         global WMST_appendDataInd;
%         Kernel = ModelAndData.Kernel;
%         
%         
%         %% Select Samples to query from 
%         % Select All of unlabeled samples for Querying
%         
% 
%         
%         
%         %samples_selected_forQuery :samples from unlabeled data to query from
%         %setQuery             :indices of query samples appended to kernel
%         
%         samples_toQuery_from = unlabeled; % Select the set to query from     
%         
%         %% WARMSTART 
%         % if warmstart is true and it's not the first time then append all of the first time append data, 
%         % don't change K matrix, so WARMSTART IS POSSIBLE FOR SCS
%         % appenddata consists of two parts
%         %                unlabeled data
%         %                labeled data which queried in active learning process( after the initial
%         %                samples in start of active learning)
%         % so appdend data indices may be labeled, and will be found in
%         % initL
%         % CAUTION: not all of unlabeled data are in appenddata, may be not all of unlabeled data is used in samples to query from, (when we use sampling method)  
%         
%         if Warmstart && WMST_beforeStart % if warmstart is true and it's the first time 
%             WMST_appendDataInd = samples_toQuery_from; 
%             WMST_beforeStart = false;
%             appendDataInd = WMST_appendDataInd;
%         elseif Warmstart                              % if warmstart is true and it's not the first time then append all of the first time append data 
%             appendDataInd = WMST_appendDataInd;
%         else                                        % if not warmstart then only append samples to query from 
%             appendDataInd = samples_toQuery_from;
%         end
%         n_a = size(appendDataInd,1);
%         nap = n+n_a; % size of appeneded data with Qset // In general Qset can be part of unlabeled data
%         n_q = size(samples_toQuery_from,1);
%         
%         labeled_appenddataind = intersect(appendDataInd,initL);
%         isLabeledappend       = false(n,1);
%         isLabeledappend(labeled_appenddataind) = true;
%         isLabeledappend       = isLabeledappend(appendDataInd);
%         %%
%         % appendData is the index of data that a copy of them appended to
%         % data for use in active learning 
%         Kqq = Kernel(appendDataInd,appendDataInd);
%         K_q = Kernel(:,appendDataInd);                  % Kernel between every thing and setQuery K(labeled and unlabeled,setQuery);
%         K   = [Kernel,K_q;K_q',Kqq];      % Kernel appended with queryset Kernels with data and with itself
%         % K   = Kernel;
%         
%         q       = sdpvar(n_q,1);        % Selection variable which we not use it directly
%         r       = sdpvar(n_u,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 
%         beta_p  = sdpvar(nap,1);        % Lagrange for alpha upper bound
%         eta_p   = sdpvar(nap,1);        % Lagrange for alpha
%         t       = sdpvar(1,1);
% 
%         setall   = 1:n;
%         setallapp= 1:nap;
%         setunlab = unlabeled;%setdiff(setall,initL);
%         settoQueryfrom = samples_toQuery_from;
%         comptoqueryfrom = setdiff(setallapp,settoQueryfrom);
%         setQuery = n+1:nap;
%         appQind  = setQuery(~isLabeledappend);
%         appNQind = setQuery(isLabeledappend);
%%        
        Klq = Klu;
        Kqq = Kuu;
        Kuq = Kuu;    
        
        K   = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
        K_l = [Kll,Klu];
        n_q = n_u;
        n_a = n_q;
        nap = n+ n_a;
        A   = [eye(3*n_q);eye(n_q),eye(n_q),eye(n_q);ones(1,n_q),zeros(1,2*n_q)];
        ONED     = [ones(n,1);zeros(n_a,1)];% ONED_i where i \in AppendLabeledData,equals 0 or 1? 
        
        %% Defining Problem
        %% Define Constraints and Objective function 
         e_nplus1 = [ones(nap+1,1);0];
        % the following matrix are for making semidefinite matrix of the
        % original problem,i.e. KV= CMat.* X, GPlus = DMat.* X
        DMat  = [ones(nap+1,nap+1),[zeros(nap,1);0];zeros(1,nap+2)];% e_nplus1*e_nplus1';
        CMat  = [K,zeros(nap,1),ones(nap,1);zeros(1,nap),0,0;ones(1,nap),0,1];
        epsconst = 0.00001;
        %% Initialization
        t     = 1;
        trange= rand;
        tavg  = rand;
        t_X   = rand;
        sdt   = rand;
        sdtrange = rand;
        sdt_X = rand;
        
        oneq   = ones(n_q,1);
        onep   = ones(n,1);
        zeroq  = zeros(n_q,1);
        zerop  = zeros(n,1);
        zeror  = zeros(n_u,1);
        oner   = ones(n_u,1);
        
        q     = rand(n_q,1);
        qrange= rand(n_q,1);
        qavg  = rand(n_q,1);
        q_qrp = qavg; 
        sdq   = rand(n_q,1);
        sdqrange = rand(n_q,1);
        sdq_X = rand(n_q,1);
        sdq_qrp = sdq_X;
        
        r     = rand(n,1);%_u,1);
        r_qrp = r;
        r_X   = rand(n,1);%_u,1);
        r_qrp   = rand(n,1);%_u,1);
        rrange= rand(n,1);%_u,1);
        ravg  = (r + r_X + r_qrp+rrange)/4;
        
        sdr   = rand(n,1);
        sdr_qrp = sdr;
        
        %sdr_s = rand(n_u,1);
        sdr_X = rand(n,1);
        sdrange = rand(n,1);
        
        pavg  = rand(n  ,1);
        p     = rand(n  ,1);
        p_X   = rand(n  ,1);
        p_qrp   = rand(n  ,1);
        prange= rand(n  ,1);
  
        
        sdp    = rand(n,1);
        sdp_s  = rand(n,1);
        sdp_X  = rand(n,1);
        sdprange = rand(n,1);
        sdp_qrp= sdp;
        
        w_o    = rand(n  ,1);
        w_oX   = rand(n  ,1);
        
        s      = rand(nap,1);
        s_X    = rand(nap,1);
        sds    = rand(nap,1);        
        
        sdw_o  = rand(n,1);
        sdqrp_u= rand(4*n_u+1,1);
        dX_X   = rand(nap+1,1);
        dX_c   = rand(nap+1,1);
        dX_c   = constraindDiag(n_l,n,nap,q,r,p,dX_c) ;% update based on the constraints
        sdX_X  = rand(nap+1,1);
        
        r_lrange= rand(n_l,1);
        r_l_X   = rand(n_l,1);
        sdr_l   = rand(n_l,1);
        
        sdres  = zeros(30,11);
        [X_lower,x_u,x_q,t_X,q_X,s_X,r_X,w_oX]= initializeX(n,n_l,nap,Yl,CMat,DMat,K_l,dX_X);
        ravg   = (r + r_X + r_qrp+ravg)/4;
        pavg   = (p + p_X + p_qrp+pavg)/4;
        qavg   = (q_X + qavg+q_qrp)/3;   
        tavg   = (tavg+trange+t_X)/3;
        rhoAUGLG = 1;
        multirho = 1.1;
        close all;
        converged = false;
        for i=1:11
            hfig(i)=figure;
        end
        repeati = 1
        while ~converged
            %% Update
            qpre   = q_qrp;
            ppre   = p_qrp;
            p_upre = p_qrp(n_l+1:n);
            rpre   = r;
            spre   = s;
            tpre   = t;
            %% TODO: we must add constraints for diagonal of X(X_ll,X_uu,X_qq)
            %% Variable consensus
            % t    : ADMMF_Xt  and ADMM_LEVENMARQUADT
            % w_oX : ADMMF_w_o and ADMM_LEVENMARQUADT
            % p    : ADMMF_w_o and ADMM_LEVENMARQUADT and ADMM_qrpu
            % s    : ADMMF_s   and ADMM_LEVENMARQUADT
            % q    : ADMM_qrpu and ADMM_LEVENMARQUADT
            % r    : ADMM_qrpu and ADMM_LEVENMARQUADT
            %% VARIABLE UPDATING STEP
            % ADMM Step for updating: t (related to  sdx_t ,x_t)
            [t]       = ADMMf_xt (tavg,sdt,lambda,rhoAUGLG);
            % ADMM Step for updating: w_o,p
            [w_o,p,r]   = ADMMf_w_o(n,n_l,K,w_oX,sdw_o,pavg,sdp,ravg,sdr,K_l,Yl,lambda_o,n_o,rhoAUGLG);
            % ADMM Step for updating: s( s=beta-eta previously )
            %                            s_X : s in (extracted from) X
            [ s ]     = ADMMf_s  (nap,n,s_X,sds,rhoAUGLG);
            % ADMM Step for updating : q, r, p_u
            %[ r ]     = ADMMf_r  (n_u,ravg,sdr,rhoAUGLG,c);
            %r = r_X;
            [ rrange] = projonfeasibler(n_l,n_u,n,ravg-sdrange);
            [ prange] = projonfeasiblep(zerop,onep,pavg-sdprange,n_o);
            [ qrange] = projonfeasibleq(zeroq ,oneq ,qavg-sdqrange,batchSize);
            [ trange] = projOnRPlus(tavg-sdtrange);
%             %[ q_qrp,p_qrp,r_qrp ];
%             q_qrp = 1-(pavg(n_l+1:n)-sdp_qrp(n_l+1:n))-(ravg-sdr_qrp);
%             p_qrp(n_l+1:n) = 1-(qavg-sdq_qrp)-(ravg-sdr_qrp);
%             r_qrp = 1-(pavg(n_l+1:n)-sdp_qrp(n_l+1:n))-(ravg-sdr_qrp);
%             

            q_qrp = 1-pavg(n_l+1:n)-ravg(n_l+1:n)-sdq_qrp;%projrange(zeror,oner,1-pavg(n_l+1:n)-ravg-sdq_qrp);
            p_qrp(n_l+1:n) = 1-qavg-ravg(n_l+1:n)-sdp_qrp(n_l+1:n);%projrange(zeror,oner,1-qavg-ravg-sdp_qrp(n_l+1:n));
            r_qrp(n_l+1:n) = 1-pavg(n_l+1:n)-qavg-sdr_qrp(n_l+1:n);%projrange(zeror,oner,1-pavg(n_l+1:n)-qavg-sdr_qrp);            
            %p_u_X     = abs(K(1:n,1:n)*w_oX);
%             b         = [ qavg-sdq;ravg-sdr_s;pavg(n_l+1:n)-sdp_s(n_l+1:n);ones(n_u,1);batchSize];
            [q_qrp,r_qrp,p_qrp] = ADMM_qrpu(n_q,n_l,n,qavg,sdq_qrp,ravg,sdr_qrp,pavg,sdp_qrp,x_u,x_q,r_X,sdqrp_u,rhoAUGLG,c);
            p_u = p_X(n_l+1:n);
            dX_c   = constraindDiag(n_l,n,nap,q,ravg,prange,dX_X-sdX_X) ;% update based on the constraints
            %% Update X parts
            % ADMM Step for updataing: 
            %       X and its parts: 
            %                   x_t (related to  sdx_t ,t)
            %                   X_h (using w_o, ?)     
            %TODO: check that scaling is properly done. 
            [X_lower,x_u,x_q,t_X,q_X,s_X,r_X,w_oX,dX_X,p_X,diffonprojX]       =...
                        ADMM_LEVENMARQUADT_X(n,n_l,n_u,nap,X_lower ,CMat, DMat,Yl,tavg,sdt_X,K_l,w_o,sdw_o,pavg,...
                        sdp_X,s,sds,qavg,sdq_X,ravg,sdr_X,dX_c,sdX_X);
            % extract X parts, X
            %
            %% LAGRANGE UPDATING STEP
            ravg   = (r + r_X + r_qrp+rrange)/4;
            pavg   = (p + p_X + p_qrp+prange)/4;
            qavg   = (    q_X + q_qrp+qrange)/3;
            tavg   = (t+trange+t_X)/3;
            sdt      = sdt      + t     -tavg;   % lagrange variable for t   == x_t
            sdtrange = sdtrange + trange-tavg;
            sdt_X    = sdt_X    + t_X   -tavg;
            
            sdw_o    = sdw_o + w_o-w_oX;% lagrange variable for w_o == w_o1
            
            sdp      = sdp      + p     -pavg;   % lagrange variable for p   == p_1
            sdp_s    = sdp_s    + p_qrp   -pavg;
            sdp_X    = sdp_X    + p_X   -pavg;
            sdprange = sdprange + prange-pavg;
            
            sds    = sds   + s  -s_X;   % lagrange variable for s   == s_X
            
            sdr    = sdr   + r  -ravg;
            sdr_X  = sdr_X + r_X-ravg;
            sdrange= sdrange+ rrange-ravg;
            
            sdq_X  = sdq_X + q_X-qavg;
            
            sdqrange = sdqrange + qrange-qavg;
            
            sdX_X  = sdX_X + dX_c - dX_X;
            
            sdq_qrp= sdq_qrp+ q_qrp-qavg;
            sdp_qrp= sdp_qrp+ p_qrp-pavg;
            sdr_qrp= sdr_qrp+ r_qrp-ravg;
            
            sdres(repeati,1) = norm(s - s_X );
            sdres(repeati,2) = norm(r  -ravg); 
            sdres(repeati,3) = norm(r_X-ravg);
            sdres(repeati,4) = norm(r_qrp-ravg);
            sdres(repeati,5) = norm(pavg(n_l+1:n)+qavg+ravg(n_l+1:n)-1);
            sdres(repeati,6) = norm(p  -pavg);
            sdres(repeati,7) = norm(p_qrp-pavg);
            sdres(repeati,8) = norm(p_X-pavg);
            sdres(repeati,9) = norm(q_X-qavg);
            sdres(repeati,10)= norm(dX_c-dX_X);
            sdres(repeati,11)= diffonprojX;
            for i=1:11
                figure(hfig(i));
                plot (sdres(1:repeati,i));
                prism;
            end
            
            %b      = [ qavg;ravg;p(n_l+1:n);ones(n_u,1);batchSize];
            %sdqrp_u= sdqrp_u   + A*[q;r_s;p_u]-b;   % ??when there are more than one multipliers for a variable? I must form MATRIX A for the constraints and update based on that.
            
            %% Rest 
            rhoAUGLG = rhoAUGLG *multirho;
            
            df = norm(s-spre)+norm(r_X-ravg)+norm(p_X-pavg);
            if df <epsconst
                converged=true;
            end
            repeati = repeati +1
        end
        %% previous
%         KVMatrix   = [K.*G_plus(setallapp,setallapp) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
%             %KVMatrix = [K.*G_plus(1:nap,1:nap) ,g_D+eta_p-beta_p;(g_D+eta_p-beta_p)',2*t/lambda];
%             cObjective = t+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r);
%             cConstraint=[beta_p>=0,   eta_p>=0,      G_plus>=0,KVMatrix>=0];
%             cConstraint=[cConstraint, sum(q)==batchSize, 0<=q,       q<=1];     % set query size to b and relax q to [0,1]
%             %if ismember(true,isLabeledappend) 
%                 %cConstraint=[cConstraint, q(isLabeledappend) ==0];
%             %end
%             cConstraint=[cConstraint, G_plus(nap+1,nap+1)==1,    G_plus(initL,nap+1)==Yl];  % G>= g g^T
%             cConstraint=[cConstraint, r>=G_plus(setunlab,nap+1), r>=-G_plus(setunlab,nap+1), r+q==1]; %r == abs(G_plus(n_l+1:n,nap+1))==abs(y_u.*(1-q))
%             cConstraint=[cConstraint, G_plus(appQind,nap+1)==q,...
%                                       G_plus(appNQind,nap+1)==0];           % constraint for g(n+1:nap)==q part 
%             cConstraint=[cConstraint, g_D(comptoqueryfrom) == 0,...%complement of the set to query from ,...     % constraints for labeled part  
%                                       g_D(settoQueryfrom)== q];
%             %TODO: use y_l for labeled appended data 
%             cConstraint=[cConstraint,diag(G_plus(initL,initL))==1,...
%                          diag(G_plus(setunlab,setunlab))<=1,...
%                          diag(G_plus(appQind,appQind))<=q,...
%                          diag(G_plus(appNQind,appNQind))==0];    
% %% Solve Problem        
%         sol = optimize(cConstraint,cObjective);
% %% Retrieve Result
%         if sol.problem==0
%             Vprimal     = value(G_plus);
%             g_Dv        = value(g_D(setunlab));
%             qinv        = value(G_plus(setQuery,nap+1));
%             qresult1     = value(q); % q Value may misguide us, because
%             %y_ui may be less than 1 or greater than -1. In these cases,
%             %q value may mislead us. 
%             % I think the best value is this value which is the nearest
%             % value to y_ui*(1-q_i): may be that's because r is not exactly
%             % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
%             % [-1,1],but we must assume it is {-1,1}
%             qyu   =value(G_plus(setunlab,nap+1));% y_u_i * (1-q_i)
%             %ra must be equal to absoulte value of qyu but in pratice the
%             %value is completely different due to relaxation of y_ui to [-1,1] 
%             ra    = value(r);
%             objpa1= value(c*sum(r));
%             objpa2= value(sum(beta_p)+sum(eta_p(setQuery)));
%             objt  = value(t);
%             cobj  = value(cObjective);
%             %sa    =value(s);
%             qresult=1-abs(qyu);
% %             tv(:,1)=g_Dv;       % 1-q
% %             tv(:,2)=qresult1;   % q   
% %             tv(:,3)=qinv;       % q in matrix G
% %             tv(:,4)=qresult;    % q in matrix G from y_u.*(1-q); 
%             [maxq,imaxq]=max(qresult);
%             qbatch=samples_toQuery_from(imaxq);
%             
%             ALresult.q = zeros(n,1);
%             ALresult.q(samples_toQuery_from) = qresult;
%             ALresult.samples_toQuery_from = samples_toQuery_from;
%             tq = k_mostlargest(qresult,batchSize);
%             ALresult.queryInd = samples_toQuery_from(tq);
%             ALresult.qBatch = zeros(n_u,1);
%             ALresult.qBatch(ALresult.queryInd) = 1;
%             
%             sum(abs(ra-qresult))
%             Y_set  = sign(value(qyu));
%             model_set =1;% for now it is set 1 to figure out it?
%         else
%             ALresult = 0;
%         end
% 
%         
%         
%         n_q = n_u;
%         Klq = Klu;
%         Kqq = Kuu;
%         Kuq = Kuu; 
%         KD  = [Kll,Klu;Klu',Kuu];    
%         K   = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];    
%         c   = 5;     %parameter for finding absolute value of y_u.*(1-q) in cost function 
%         n   = n_l+n_u;
%         nap   =n_l+n_u+n_q;
%         
%         p     =sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
%         w_o   =sdpvar(n,1);          % For w_o function 
%         G_plus     =sdpvar(nap+1,nap+1);  
%         q     =sdpvar(n_u,1);        % Selection variable which we not use it directly
%         r     =sdpvar(n_u,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 
%         beta_p=sdpvar(nap,1);        % Lagrange for alpha upper bound
%         eta_p =sdpvar(nap,1);        % Lagrange for alpha lower bound
%         t     =sdpvar(1,1);          % 
%         % Unlabeled data is from 1 to n_l
%         initL =[1:n_l]; 
%         setall   =1:n;
%         setunlab =setdiff(setall,initL);
%         setQuery =n+1:nap;
%         
%         %y_l     =yapp(initL);
%         
%         
%         KVMatrix     = sdpvar(nap+1,nap+1);
%         g_D     =sdpvar(nap,1);
%         rl    =sdpvar(n_l,1);
%         Pu    =sdpvar(n_u,1); % a variable for Y_u \Phi(X_u)^T w_o
%         Yu    =sdpvar(n_u,1);
%         KVMatrix = [K.*G_plus(1:nap,1:nap) ,1-g_D+eta_p-beta_p;(1-g_D+eta_p-beta_p)',2*t/lambda];
%         
%         
%         cConstraint=[beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
%         cConstraint=[cConstraint,sum(q)==batchSize,0<=q,q<=1];% constraints on q 
%         % constraints on G_plus       
%         cConstraint=[cConstraint,G_plus(nap+1,nap+1)==1,...
%                                  G_plus(initL,nap+1)==rl];
%         % it is better to substitute p with y_l.*w_o^T\phi(x_i)
%         cConstraint=[cConstraint,diag(G_plus(1:n_l,1:n_l))==1-p(1:n_l),...
%                      diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))==r,...% change inquality to equality in this line
%                      diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))==q];  % change inquality to equality in this line(last constraint)
%         cConstraint=[cConstraint,diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))<=1-p(n_l+1:n),...
%                      diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))<=1-q];
%                  
%         cConstraint=[cConstraint,G_plus(n+1:nap,nap+1)==q,...
%                      g_D(initL)==zeros(n_l,1),g_D(n+1:nap)==zeros(n_q,1),g_D(setunlab)==1-r];
%                  
%         
%                  
%         cConstraint=[cConstraint,-p<=KD*w_o<=p,p<=1,rl==Yl-KD(1:n_l,:)*w_o];
%         % for absolute value of y_u.*(1-pu).*(1-q)
%         cConstraint=[cConstraint,r>=G_plus(n_l+1:n,nap+1),r>=-G_plus(n_l+1:n,nap+1)];
%         cConstraint=[cConstraint,r+q+p(n_l+1:n)==1];
%         
%         
%         formp = 1;
%         if formp==1
%          n_o      = 4;    
%         % Form 1: sum(p)<n_o, as a constraint
%         cConstraint=[cConstraint,sum(p)<=n_o];
%         %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
%         cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);
%         
%         else
%              n_o      = 4;
%             cp       =2;
%         % Form 2: c*sum(p)+Objective as an addition to objective not as a
%         cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r)+cp*sum(p); %addition of sum(q) didn't help+3*sum(q);
%         
%         end
%%
%         cConstraint=[cConstraint,Pu<=KD(n_l+1:n_l+n_u,:)*w_o-Yu+1,...
%                                  Pu<=KD(n_l+1:n_l+n_u,:)*w_o+2*p(n_l+1:n_l+n_u),...
%                                  Pu<=Yu+1-KD(n_l+1:n_l+n_u,:)*w_o,...
%                                  Pu<=2*p(n_l+1:n_l+n_u)-KD(n_l+1:n_l+n_u,:)*w_o];
%         cConstraint=[cConstraint,Pu>=KD(n_l+1:n_l+n_u,:)*w_o+Yu-1,...
%                                  Pu>=KD(n_l+1:n_l+n_u,:)*w_o-2*p(n_l+1:n_l+n_u),...
%                                  Pu>=-Yu-1-KD(n_l+1:n_l+n_u,:)*w_o,...
%                                  Pu>=-KD(n_l+1:n_l+n_u,:)*w_o-2*p(n_l+1:n_l+n_u)];                    
%% 
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))+diag(V(setQuery,setQuery))==ones(n_q,1)];
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))==diag(V(setQuery,setQuery))-2*q(setunlab)+ones(n_q,1)];
        
%         sol = optimize(cConstraint,cObjective);
%         if sol.problem==0
%             Vprimal     = value(G_plus);
%             qv     = value(q); % q Value may misguide us, because
%             pv     = value(p);
%             g_Dv   = value(g_D(n_l+1:n));
%             qinv   = value(G_plus(n_l+n_u+1:nap,nap+1));
%             qresult1     = value(q);
%             %Puv    = value(Pu);
%             w_ov    = value(w_o);
%             %Yuv    = value(Yu);
%             sv     = value(g_D);
%             rlv    = value(rl);
%             beta_pv= value(beta_p);
%             eta_pv = value(eta_p);
%             rv     = value(r);
%             G_data = value(G_plus(1:n,1:n));
%             
%             %y_ui may be less than 1 or greater than -1. In these cases,
%             %q value may mislead us. 
%             % I think the best value is this value which is the nearest
%             % value to y_ui*(1-q_i): may be that's because r is not exactly
%             % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
%             % [-1,1],but we must assume it is {-1,1}
%             qyu   =value(G_plus(n_l+1:n,nap+1));% y_u_i * (1-q_i)
%             %ra must be equal to absoulte value of qyu but in pratice the
%             %value is completely different due to relaxation of y_ui to [-1,1] 
%             %sa    =value(s);
%             if formp==1
%             qresult=1-abs(qyu);
%             qresult=qresult.*(1-pv(n_l+1:n));
%             else
%             qresult=qv;    
%             end
%             
%             tv(:,1)=g_Dv;       % (1-q).*(1-pu)
%             tv(:,2)=qresult1;   % q   
%             tv(:,3)=qinv;       % q in matrix G
%             tv(:,4)=qresult;    % q in matrix G from y_u.*(1-q).*(1-pu); 
%             
%             [maxq,imaxq]=max(qresult);
%             qbatch=imaxq;
%             Y_set  = sign(value(qyu));
%             % set learning parameters 
%             model_set.beta_p = beta_pv;
%             model_set.beta_p = eta_pv;
%             model_set.w_oxT_i= w_ov'*KD;
%             model_set.G      = G_data;
%             model_set.g      = value(1-g_D);
%             
%         end
end
function [X_lower,x_u,x_q,t_X,q_X,s_X,r_X ,w_oX] = initializeX(n,n_l,nap,Yl,CMat,DMat,Klu,dX)
        X_yl   = Yl*Yl';
        X_lower= rand((nap+2)*(nap+3)/2,1);
        
        % expand to full size matrix
        nx  = nap + 2;
        b   = tril(ones(nx));
        b(b == 1) = X_lower;
        X   = b;
        X   = (X + X');
        X   = X - diag(diag(X)) / 2;
        z   = eye(nap+1);
        X(z==1) = dX;
        X ( 1:n_l,1:n_l ) = X_yl;
        X = proj_ontwo_cones(nap,X,CMat,DMat);
        X(eye(nap+2) == 1) = X(eye(nap+2) == 1) ./ sqrt(2);
        [t_X,q_X,s_X,r_X,x_u,x_q,w_oX,x_l] = extractXparts(n,nap,n_l,X,Klu,Yl);
        X_lower = X(tril(ones(nap+2)) == 1);
end
function [dX_c]=constraindDiag(n_l,n,nap,q,r,p,dX_c) 
    FMin        = [1-q,1-p(n_l+1:n),r(n_l+1:n)];
    diagX       = [1-p(1:n_l);min(FMin,[],2)];
    diagX       = max(diagX,0);
    
    dX_c        = projrange(zeros(n,1),diagX,dX_c(1:n));
    dX_c(n+1:nap) = max(q,0);
    dX_c(nap+1)   = 1;
end
function [X_lower,x_u,x_q,t_X,q_X,s_X,r_X,w_oX,dX_X,p_X,diffonprojX]  = ADMM_LEVENMARQUADT_X(n,n_l,n_u,nap,X_lower ,CMat, DMat,Yl,t,sdt,K_l,...
                                                                                w_o,sdw_o,pavg,sdp_X,s,sds,q,sdq,ravg,sdr_X,...
                                                                                dX_c,sdX_X)
% X : I_sdp(CMat.*X)+I_sdp(DMat.*X)
%% Make it Full Matrix
nx = nap + 2;
bs = tril(ones(nx));
bs(bs == 1) = X_lower;
X = bs;
X = (X + X');
X = X - diag(diag(X)) / 2;
%% store X parts 
% t-> t+sdt,
% s->s+sds,r->ravg-sdr,q->q+sdq,p->pavg-sdp_X,w_o->w_o+sdw_o,dX->dX_c+sdX_X
[X] = storeXparts(n,n_l,nap,n_u,X,Yl,w_o+sdw_o,K_l,q+sdq,s+sds,ravg-sdr_X,t+sdt,pavg-sdp_X,dX_c+sdX_X);

%% Project onto Cones
X1 = proj_ontwo_cones(nap,X,CMat,DMat);
diffonprojX  = norm(X1-X,'fro');
X = X1;
[t_X,q_X,s_X,r_X,x_u,x_q,w_oX,x_l,dX_X] = extractXparts(n,nap,n_l,X,K_l,Yl);
p_X = projrange(zeros(n,1),[1-x_l;1-x_u],pavg-sdp_X);
%% Extract X parts
X(eye(nap+2) == 1) = X(eye(nap+2) == 1) ./ sqrt(2);
X_lower = X(tril(ones(nap+2)) == 1);
% make 1 and zero cells 
%X (nap+1,nap+1) =1;
X (nap+2,nap+1) =0;
X (nap+1,nap+2) =0;
%% Extract Lower Triangle
% scale diags by 1/sqrt(2)

end
function [t_X,q_X,s_X,r_X,x_u,x_q,w_oX,x_l,dX_X] = extractXparts(n,nap,n_l,X,K_l,y_l)
%% Extract X parts
%r_l = y_l-K_l*w_o;
Kwo = -X (1:n_l,nap+1 )+y_l;
w_oX= K_l\Kwo;
w_ovar = sdpvar(n,1);
cObjective = norm(K_l*w_ovar-Kwo,2)^2;
sol=optimize([],cObjective);
if sol.problem==0
    w_oX = value(w_ovar);
else
    w_oX = zeros(n,1);
end
%X (nap+1,1:n_l ) = r_l;
q_X = X (n+1:nap,nap+1  );
t_X = X (nap+2,nap+2);
v_X = X (n_l+1:n,nap+1);
r_X = [X(1:n_l,nap+1);abs(v_X)];
s_X(1:n_l,1) = 1-X (1:n_l,nap+2 );
g_D                  = 1-r_X(n_l+1:n);
s_X(n_l+1:n,1)         = 1- X(n_l+1:n,nap+2)-g_D;
s_X(n+1:nap,1)         = -X (nap+2  ,n+1:nap );

x_u = diag(X(n_l+1:n,n_l+1:n));
x_l = diag(X(1:n_l,1:n_l));
x_q = diag(X(n+1:nap,n+1:nap));
dX_X = diag(X(1:nap+1,1:nap+1));
end
function [X] = storeXparts(n,n_l,nap,n_u,X,y_l,w_o,K_l,q,s,r,t,p,dX)
X_yl = y_l*y_l';
z = zeros(nap+2);
z (1:nap+1,1:nap+1)   = eye(nap+1);
X (z == 1 )   = dX;%ap
%X ( 1:n_l,1:n_l )   = X_yl;
r_l                 = r(1:n_l);%y_l-K_l*w_o;
r_l = projrange(-ones(n_l,1),ones(n_l,1),r_l);
X (1:n_l,nap+1 )    = r_l;
X (nap+1,1:n_l )    = r_l;
fortest=1;
if fortest==1
X (n+1:nap,nap+1  ) = q;
X (nap+1  ,n+1:nap) = q;
X (nap+1  ,nap+1  ) = 1;
end
X (1:n_l,nap+2 ) = 1-s(1:n_l);
X (nap+2,1:n_l ) = 1-s(1:n_l);

g_D                  = 1-r(n_l+1:n);
X (n_l+1:n,nap+2   ) = 1-g_D-s(n_l+1:n);
X (nap+2  ,n+1:nap ) = -s(n+1:nap);
X (nap+2  ,nap+2   ) = t; 
% make 1 and zero cells 
%X (nap+1,nap+1) =1; it will be imposed based on the AUGLG method
X (nap+2,nap+1) =0;
X (nap+1,nap+2) =0;

end
function [t] = ADMMf_xt (x_tpre,sdx_tpre,lambda,rhoAUGLG)
% ADMM step for t: it only updates t based on the value of x_t
% t = argmin_t rhoAUGLG* lambda/2* t +1/2 \Vert t-x_tpre+sdx_tpre \Vert^2
tvar = sdpvar(1,1);
cObjective  = rhoAUGLG* lambda/2* tvar + 1/2* norm( tvar-x_tpre+sdx_tpre,2)^2; 
cConstraint = [];%tvar>0];
sol = optimize( cConstraint,cObjective);
if sol.problem == 0
    t = value(tvar);
else
    assert('cannot solve for t');
end
end
function [w_o,p,r]   = ADMMf_w_o(n,n_l,K,w_oX,sdw_o,pavg,sdp,r,sdr,K_l,Yl,lambda_o,n_o,rhoAUGLG)
% ADMM step for w_o,p: it updates w_o and p based on the values of w_o1,p_1
% [w_o,p]= argmin_{[w_o,p]} rhoAUGLG* lambda_o*\Vert w_o \Vert^2 + 1/2* \Vert w_o-w_o1+sdw_o \Vert^2 + 1/2* \Vert p-p_1+sdp \Vert^2
%                s.t. sum(p)<= n_o && p<=1, p>=\vert w_o^t\Phi(x_i)\vert 
%r(n_l+1:n)=r;
secondrun=0;
rl   = r(1:n_l)-sdr(1:n_l);
w_ovar = sdpvar(n,1);
pvar   = sdpvar(n,1);
cConstraint = [-pvar<=K(1:n,1:n)*w_ovar<=pvar];%,pvar<=1];%, sum(pvar) <= n_o];
if secondrun==1
cConstraint = [cConstraint,rl==Yl-K_l*w_ovar];
end;
cObjective  = rhoAUGLG*lambda_o/2*norm(w_ovar,2)^2+1/2* norm(w_ovar-w_oX+sdw_o,2)^2+1/2*norm(pvar-pavg+sdp,2)^2;
sol = optimize(cConstraint,cObjective);
if sol.problem == 0
   w_o  = value(w_ovar);
   p    = value(pvar);
   r(1:n_l)=value(Yl-K_l*w_ovar);
else
    assert(true,'cannot solve for w_o,p');
end

end
function [ s ]     = ADMMf_s  (nap,n,s_X,sds,rhoAUGLG)
% ADMM step for s=beta-eta (in previous formulation) it updates s based on
%               s_X and sds
% s = argmin_s rhoAUGLG*(\sum_{i in D) (s_i)_+ + \Vert s_Q \Vert_1 ) + 1/2 * \Vert s-s_X+sds \Vert^2 
svar = sdpvar(nap,1);
cConstraint = [];
cObjective  = rhoAUGLG*sum(max(svar(1:n),0))+norm(svar(n+1:nap),1)+ 1/2* norm(svar-s_X+sds,2)^2;
sol = optimize(cConstraint,cObjective);
if sol.problem ==0
    s = value(svar);
else
    assert(' cannot solve for s');
end

end
function [q,r_s,p_qrp] = ADMM_qrpu(n_q,n_l,n,qavg,sdq,ravg,sdr_s,pavg,sdp_s,x_u,x_q,r_X,sdqrp_u,rhoAUGLG,c)
% ADMM step for q, r, p_u 
%               it updates based on q_X,r_X,p_u_X,sdq,sdr,sdp_u,sdpuqr_sumone,sdqsum_one
%  consider objective + c*sum(r_i)
npr = n_q;
p_u = pavg(n_l+1:n);
% solve for q,and project on constraints:x_q <= qvar; 0<=qvar;qvar<=1;x_u<=1-qvar
qvar        = sdpvar(npr,1);
rvar        = sdpvar(n,1);
puvar       = sdpvar(n,1);
cConstraint = [qvar+rvar(n_l+1:n)+puvar(n_l+1:n)==1];
%cConstraint = [qvar>=0;qvar<=1];
%cConstraint = [cConstraint; 0 <= rvar;rvar<=1];%; v_X<=rvar;-v_X<=rvar];%x_u <= rvar; v_X<=rvar;-v_X<=rvar];
%cConstraint = [cConstraint; 0<=puvar,puvar<=1];
cObjective  = norm([qvar;rvar;puvar]-[qavg-sdq;ravg-sdr_s;pavg-sdp_s],2)^2;%+norm(rvar-ravg+sdr_s,2)^2+norm(puvar-p_u+sdp_s(n_l+1:n),2)^2;
sol         = optimize(cConstraint,cObjective);%cConstraint
if sol.problem ==0
    q    = value(qvar);
    p_qrp = value(puvar);
    r_s  = value(rvar);
else
    assert(' cannot solve for q');
end
%% 
% project on constraint: x_q<= q
% ind    = x_q > q;
% q(ind) = x_q(ind); 
% 
% ind    = x_u > (1-q);
% q(ind) = 1-x_u(ind); 
% 
% % project on constraint: 0<= q
% ind    = q < 0;
% q(ind) = 0; 
% % project on constraint: q <= 1
% ind    = q > 1;
% q(ind) = 1;
% project on constraint: x_q<= q
% solve for r,

% cConstraint = [x_u <= rvar; v_X<=rvar;-v_X<=rvar];
% cObjective  = rhoAUGLG*c*sum(rvar)+1/2*norm(A*[q;rvar;p_u]-b+sdqrp_u,2);
% sol = optimize(cConstraint,cObjective);
% if sol.problem ==0
%     r= value(rvar);
% else
%     assert(' cannot solve for q');
% end
% % solve for p_u,
% 
% cConstraint = [0<=puvar,puvar<=1 ];
% cObjective  = 1/2*norm(A*[q;r;puvar]-b+sdqrp_u,2);
% sol = optimize(cConstraint,cObjective);
% if sol.problem ==0
%     p_EXu= value(puvar);
% else
%     assert(' cannot solve for q');
% end
% ind = p_EXu <0;
% p_EXu(ind) = 0;
% ind = p_EXu >1;
% p_EXu(ind) = 1;
%%
%p_qrp = [pavg(1:n_l);p_EXu];
end
function X         = proj_ontwo_cones(nap,X,CMat,DMat)
% This function projects X.*CMat and X.*DMat to sdp cone. First for CMat
% and then for DMat. 
% CMat(:,nap+2)=CMat(nap+2,:)=0, and other cells are one
% DMat(:,nap+1)=DMat(nap+1,:)=0
% instead of multipliying with DMat, we use just elements 1:nap+1,1:nap+1
[V,S] = eig(X([1:nap,nap+2],[1:nap,nap+2]).*CMat([1:nap,nap+2],[1:nap,nap+2]));
S     = diag(S);
idx   = find(S>0);
V     = V(:,idx);
S     = S(idx);
C= CMat;
XCMat = V*diag(S)*V';
MIN_SCALE = 1e-4;
MAX_SCALE = 1e4;
minScale = MIN_SCALE * sqrt(nap^2);
maxScale = MAX_SCALE * sqrt(nap^2);
CMat(CMat<minScale)=1;
X([1:nap,nap+2],[1:nap,nap+2])  = XCMat./CMat([1:nap,nap+2],[1:nap,nap+2]);
% XLc   = sdpvar(nap+1,nap+1);
% c     = 2;
% cObjective = norm(XLc.*CMat([1:nap,nap+2],[1:nap,nap+2])-XCMat,1);%+c*norm(XLc,'fro')^2;
% sol = optimize([],cObjective);
% if sol.problem == 0 
%     XLU= value(XLc);
%     XDi=XLU-XCMat;
%     norm(XDi,'fro')
%     X([1:nap,nap+2],[1:nap,nap+2]) = value(XLc); 
% else 
%     assert(true,'cannot solve for X');
% end 
norm(X([1:nap,nap+2],[1:nap,nap+2]).*CMat([1:nap,nap+2],[1:nap,nap+2]) -XCMat,'fro')
X   = (X + X')/2;

%X = XL;
[V,S] = eig(X(1:nap+1,1:nap+1));
S = diag(S);
% 
% ns = size(S,1);
% lam = sdpvar(ns,1);
% Z   = sdpvar(nap+1,nap+1,'full');
% 
% cConstraint=[S+lam >=0, Z==V*diag(S+lam)*V',Z(nap+1,nap+1)==1];
% cObjective = norm(X(1:nap+1,1:nap+1)-Z,'fro');
% sol = optimize(cConstraint,cObjective);
% if sol.problem==0
%     Xp = V*diag(S+value(lam))*V;
%     norm(Xp-X(1:nap+1,1:nap+1),'fro')
%     
% end
% Xp=sdpvar(nap+1,nap+1);
% cConstraint=[Xp>=0,Xp(nap+1,nap+1)==1];
% cObjective = norm(X(1:nap+1,1:nap+1)-Xp,'fro');
% sol = optimize(cConstraint,cObjective);
% if sol.problem==0
%     Xpv = value(Xp);
%     norm(Xpv-X(1:nap+1,1:nap+1),'fro')/norm(X(1:nap+1,1:nap+1),'fro')
%     
% end
idx = find(S>0);
V = V(:,idx);
S = S(idx);
XDMat = V*diag(S)*V';
Xscale=XDMat(nap+1,nap+1);
XDMat = XDMat./Xscale;
norm(XDMat-X(1:nap+1,1:nap+1),'fro')/norm(X(1:nap+1,1:nap+1),'fro')
X(1:nap+1,1:nap+1) = XDMat; 
X   = (X + X')/2;
end
function [ r ]     = ADMMf_r  (n_u,ravg,sdr,rhoAUGLG,c)
rvar       = sdpvar(n_u,1);
% when rvar, can be negative, the first term, sum becames negative and the
% solution diverges. Given rvar>0, so we added max(rvar,0)
cObjective = c* norm(rvar,1) + 1/(2*rhoAUGLG)*norm(rvar-ravg+sdr,2)^2;
cConstraint= [];%rvar>=0];
sol = optimize(cConstraint,cObjective);
if sol.problem == 0
   r  = value(rvar);
else
    assert(true,'cannot solve for r');
end
end
function [ varib ]  = projrange(l,u,varib)
% if u is less than l in some elements, project them first
isbel  = u < l;
u(isbel)= l(isbel);

isbel  = varib < l;
varib (isbel) = l(isbel);
isabo  = varib >= u;
varib (isabo) = u(isabo);
end
function [ qrange] = projonfeasibleq(zeroq ,oneq ,q,batchSize)
    %[ qrange] = projrange(zeroq ,oneq ,q);
    qrange    = ProjectOntoSimplex(q,batchSize);
end
function [ prange] = projonfeasiblep(zerop,onep,pavg,n_o)
    [ prange] = projrange(zerop,onep,pavg);
    if sum(prange)>n_o
        prange    = ProjectOntoSimplex(prange,n_o);
    end
end
function [ t] = projOnRPlus(t)
if t < 0
    t = 0;
end
end
function [ rrange] = projonfeasibler(n_l,n_u,n,r)
    rrange(n_l+1:n,1)= projrange(zeros(n_u,1),ones(n_u,1),r(n_l+1:n));
    rrange(1:n_l  ,1)= projrange(-ones(n_l,1),ones(n_l,1),r(1:n_l));
end