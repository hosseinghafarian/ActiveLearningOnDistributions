function [ALresult,model_set,Y_set]= ActiveDConvexRelax1(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda)
        % code of this function before considering warmstart 
        % Aassume all of unlabeled data are candidates for active learning query points    
        n_q = n_u;
        Klq = Klu;
        Kqq = Kuu;
        Kuq = Kuu;    
        K   = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];    
        c   = 5;     %parameter for finding absolute value of y_u.*(1-q) in cost function 
        n   = n_l+n_u;
        nap   =n_l+n_u+n_q;
        G_plus     =sdpvar(nap+1,nap+1);  
        q     =sdpvar(n_u,1);        % Selection variable which we not use it directly
        r     =sdpvar(n_u,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 
        beta_p=sdpvar(nap,1);        % Lagrange for alpha upper bound
        eta_p =sdpvar(nap,1);        % Lagrange for alpha
        t     =sdpvar(1,1);
        % Unlabeled data is from 1 to n_l
        initL =[1:n_l]; 
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        setQuery =n+1:nap;
        ONED     = [ones(n,1);zeros(n_q,1)];
        KVMatrix     = sdpvar(nap+1,nap+1);
        g_D     =sdpvar(nap,1);
        KVMatrix = [K.*G_plus(1:nap,1:nap) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
        %KVMatrix = [K.*G_plus(1:nap,1:nap) ,g_D+eta_p-beta_p;(g_D+eta_p-beta_p)',2*t/lambda];
        cObjective = t+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);
        
        cConstraint=[beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0];
        cConstraint=[cConstraint,sum(q)==batchSize,0<=q,q<=1];
        cConstraint=[cConstraint,G_plus(nap+1,nap+1)==1,G_plus(initL,nap+1)==Yl];
        cConstraint=[cConstraint,r>=G_plus(n_l+1:n,nap+1),r>=-G_plus(n_l+1:n,nap+1),r+q==1];
        %cConstraint=[cConstraint,G_plus(n+1:nap,nap+1)==q,...
                     %g_D(initL)==zeros(n_l,1),g_D(n+1:nap)==zeros(n_q,1),g_D(setunlab)==q];
        cConstraint=[cConstraint,G_plus(n+1:nap,nap+1)==q,...
                     g_D(initL)==zeros(n_l,1),g_D(n+1:nap)==0,g_D(setunlab)==q];
        cConstraint=[cConstraint,diag(G_plus(1:n_l,1:n_l))==1,...
                     diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))<=1,...
                     diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))<=q];

        % When both of the following constraints are added it seems that
        % more reasonable point is queryied(in single point query,Batch
        % mode not tested)
        % cConstraint=[cConstraint,diag(V(setunlab,setunlab))+diag(V(setQuery,setQuery))==ones(n_q,1)];
        % cConstraint=[cConstraint,diag(V(setunlab,setunlab))==diag(V(setQuery,setQuery))-2*q+ones(n_q,1)];
        
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            G_data     = value(G_plus);
            g_Dv        = value(g_D(n_l+1:n));
            qinv        = value(G_plus(n_l+n_u+1:nap,nap+1));
            qresult1    = value(q); % q Value may misguide us, because
            betval      = value(beta_p);
            etval       = value(eta_p);
            KVVAL       = value(KVMatrix);
            %y_ui may be less than 1 or greater than -1. In these cases,
            %q value may mislead us. 
            % I think the best value is this value which is the nearest
            % value to y_ui*(1-q_i): may be that's because r is not exactly
            % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
            % [-1,1],but we must assume it is {-1,1}
            qyu   =value(G_plus(n_l+1:n,nap+1));% y_u_i * (1-q_i)
            %ra must be equal to absoulte value of qyu but in pratice the
            %value is completely different due to relaxation of y_ui to [-1,1] 
            ra    = value(r);
            objpa1= value(c*sum(r));
            objpa2= value(sum(beta_p)+sum(eta_p(n+1:nap)));
            objt  = value(t);
            cobj  = value(cObjective);
            %sa    =value(s);
            qresult=1-abs(qyu);
            ALresult.q = zeros(n,1);
            ALresult.q(samples_toQuery_from) = qresult;
            ALresult.samples_toQuery_from = samples_toQuery_from;
            tq = k_mostlargest(qresult,batchSize);
            ALresult.queryInd = samples_toQuery_from(tq);
            ALresult.qBatch = zeros(n_u,1);
            ALresult.qBatch(ALresult.queryInd) = 1;
            
            %sum(abs(ra-qresult))
            Y_set  = sign(value(qyu));
            model_set.beta_p = value(beta_p);
            model_set.eta_p  = value(eta_p);
            model_set.G      = G_data;
            model_set.g      = 1-g_Dv;
            %sum(abs(ra-qresult))
        end
end