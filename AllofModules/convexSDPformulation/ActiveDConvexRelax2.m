function [ALresult,model_set,Y_set]= ActiveDConvexRelax2(WMST_beforeStart, Model, learningparams)
        global cnstData
        ALresult.active  = true;
        initL     = cnstData.initL;
        unlabeled = cnstData.unlabeled;
        n_l       = cnstData.n_l;
        n_u       = cnstData.n_u;
        batchSize = cnstData.batchSize;
        n_o       = cnstData.n_o;
        % code with WARMSTART 
        % Aassume all of unlabeled data are candidates for active learning query points    
        % n_l :size(initL,2);
        % n_u :n-n_l;
        % n_q :n_u;
        norm1_form = 1;
        
        global Warmstart;
        global WMST_appendDataInd;
        assert(Warmstart,'This function works only Warmstart is active');
        Kernel = ModelAndData.Kernel;
        
        c   = 5;           % parameter for finding absolute value of y_u.*(1-q) in cost function 
        
        %% Select Samples to query from 
        % Select All of unlabeled samples for Querying
        
        n   = n_l+n_u;     % size of data 
        
        
        %samples_selected_forQuery :samples from unlabeled data to query from
        %setQuery             :indices of query samples appended to kernel
        
        samples_toQuery_from = unlabeled; % Select the set to query from     
        
        %% WARMSTART 
        % if warmstart is true and it's not the first time then append all of the first time append data, 
        % don't change K matrix, so WARMSTART IS POSSIBLE FOR SCS
        % appenddata consists of two parts
        %                unlabeled data
        %                labeled data which queried in active learning process( after the initial
        %                samples in start of active learning)
        % so appdend data indices may be labeled, and will be found in
        % initL
        % CAUTION: not all of unlabeled data are in appenddata, may be not all of unlabeled data is used in samples to query from, (when we use sampling method)  
        
        if Warmstart && WMST_beforeStart % if warmstart is true and it's the first time 
            WMST_appendDataInd = samples_toQuery_from; 
            WMST_beforeStart = false;
            appendDataInd = WMST_appendDataInd;
        elseif Warmstart                              % if warmstart is true and it's not the first time then append all of the first time append data 
            appendDataInd = WMST_appendDataInd;
        else                                        % if not warmstart then only append samples to query from 
            appendDataInd = samples_toQuery_from;
        end
        n_a = size(appendDataInd,1);
        nap = n+n_a; % size of appeneded data with Qset // In general Qset can be part of unlabeled data
        n_q = size(samples_toQuery_from,1);
        
        labeled_appenddataind = intersect(appendDataInd,initL);
        isLabeledappend       = false(n,1);
        isLabeledappend(labeled_appenddataind) = true;
        isLabeledappend       = isLabeledappend(appendDataInd);
        %ylabeled_append       = Yl(labeled_appenddataind);
        %%
        % appendData is the index of data that a copy of them appended to
        % data for use in active learning 
        Kqq = Kernel(appendDataInd,appendDataInd);
        K_q = Kernel(:,appendDataInd);                  % Kernel between every thing and setQuery K(labeled and unlabeled,setQuery);
        K   = [Kernel,K_q;K_q',Kqq];      % Kernel appended with queryset Kernels with data and with itself
        % K   = Kernel;
        
        G_plus  = sdpvar(nap+1,nap+1);  
        q       = sdpvar(n_q,1);        % Selection variable which we not use it directly
        r       = sdpvar(n_u,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 

        if norm1_form == 1
            beta_p  = sdpvar(nap,1);        % Lagrange for alpha upper bound
            eta_p   = sdpvar(nap,1);        % Lagrange for alpha
        elseif norm1_form == 2
            uDpart  = sdpvar(n,1);
            uQpart  = sdpvar(n_a,1);
            uall    = [uDpart;uQpart];
        end        
        t       = sdpvar(1,1);
        % Unlabeled data is from 1 to n_l
%         initL    = 1:n_l; 
        setall   = 1:n;
        setallapp= 1:nap;
        setunlab = unlabeled;%setdiff(setall,initL);
        settoQueryfrom = samples_toQuery_from;
        comptoqueryfrom = setdiff(setallapp,settoQueryfrom);
        setQuery = n+1:nap;
        appQind  = setQuery(~isLabeledappend);
        appNQind = setQuery(isLabeledappend);
        %y_l     =yapp(initL);
       
        KVMatrix = sdpvar(nap+1,nap+1);
        g_D      = sdpvar(nap,1);
        ONED     = [ones(n,1);zeros(n_a,1)];% ONED_i where i \in AppendLabeledData,equals 0 or 1? 
        % Caution: Don't forget ONED 

        % Wow!. If you set g_D=1-q,  you get nothing. May be because of the
        % numerical errors
        %% Defining Problem
        %% LATEX Problem 
        % 
        % $$\begin{alignat}{3}&\min_{q,y_u} \min_{w_o,p}\min_{\beta,\eta}
        % &&t-\eta^\mathsf{T} lo+\beta^\mathsf{T} up\notag\\
        % &\textrm{\lr{s.t.}}  &&\beta ,\eta \geq 0,y_{ui}\in [-1,1], q_i\in [0,1]\notag\\
        %&&&\begin{bmatrix}
        %K\odot G &g_D+\eta-\beta\\
        %(g_D+\eta-\beta)^\mathsf{T}&\frac{2}{\lambda} t
        %\end{bmatrix} \succcurlyeq 0\cr 
        %&&& \left[\begin{array}{cccc}
        %\multicolumn{3}{c}{\multirow{3}{*}{\Huge$G$}} & y_l \\
        % & & &r \\
        % & & &q\\
        %y_l &r& q & 1\\
        %\end{array}\right] \succcurlyeq 0\cr
        %&&&\vert r \vert + q = 1,\sum_{i\in Q} q_i = b
        %\end{alignat}$$
        % 
        %% Define Constraints and Objective function 
        if norm1_form == 1
            KVMatrix   = [K.*G_plus(setallapp,setallapp) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
            %KVMatrix = [K.*G_plus(1:nap,1:nap) ,g_D+eta_p-beta_p;(g_D+eta_p-beta_p)',2*t/lambda];
            cObjective = t+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r);
            cConstraint=[beta_p>=0,   eta_p>=0,      G_plus>=0,KVMatrix>=0];
            cConstraint=[cConstraint, sum(q)==batchSize, 0<=q,       q<=1];     % set query size to b and relax q to [0,1]
            %if ismember(true,isLabeledappend) 
                %cConstraint=[cConstraint, q(isLabeledappend) ==0];
            %end
            cConstraint=[cConstraint, G_plus(nap+1,nap+1)==1,    G_plus(initL,nap+1)==Yl];  % G>= g g^T
            cConstraint=[cConstraint, r>=G_plus(setunlab,nap+1), r>=-G_plus(setunlab,nap+1), r+q==1]; %r == abs(G_plus(n_l+1:n,nap+1))==abs(y_u.*(1-q))
            cConstraint=[cConstraint, G_plus(appQind,nap+1)==q,...
                                      G_plus(appNQind,nap+1)==0];           % constraint for g(n+1:nap)==q part 
            cConstraint=[cConstraint, g_D(comptoqueryfrom) == 0,...%complement of the set to query from ,...     % constraints for labeled part  
                                      g_D(settoQueryfrom)== q];
            %TODO: use y_l for labeled appended data 
            cConstraint=[cConstraint,diag(G_plus(initL,initL))==1,...
                         diag(G_plus(setunlab,setunlab))<=1,...
                         diag(G_plus(appQind,appQind))<=q,...
                         diag(G_plus(appNQind,appNQind))==0];    
        elseif norm1_form == 2
                        KVMatrix   = [K.*G_plus(setallapp,setallapp) ,ONED-g_D-uall;(ONED-g_D-uall)',2*t/lambda];
            %KVMatrix = [K.*G_plus(1:nap,1:nap) ,g_D+eta_p-beta_p;(g_D+eta_p-beta_p)',2*t/lambda];
            cObjective = t+norm(uQpart,1)+ sum(max(uDpart,0))+c*sum(r);
            cConstraint=[G_plus>=0,KVMatrix>=0];
            cConstraint=[cConstraint, sum(q)==batchSize, 0<=q,       q<=1];     % set query size to b and relax q to [0,1]
            %if ismember(true,isLabeledappend) 
                %cConstraint=[cConstraint, q(isLabeledappend) ==0];
            %end
            cConstraint=[cConstraint, G_plus(nap+1,nap+1)==1,    G_plus(initL,nap+1)==Yl];  % G>= g g^T
            cConstraint=[cConstraint, r>=G_plus(setunlab,nap+1), r>=-G_plus(setunlab,nap+1), r+q==1]; %r == abs(G_plus(n_l+1:n,nap+1))==abs(y_u.*(1-q))
            cConstraint=[cConstraint, G_plus(appQind,nap+1)==q,...
                                      G_plus(appNQind,nap+1)==0];           % constraint for g(n+1:nap)==q part 
            cConstraint=[cConstraint, g_D(comptoqueryfrom) == 0,...%zeros(nap-n_q,1),...     % constraints for labeled part  
                                      g_D(settoQueryfrom)== q];
            %TODO: use y_l for labeled appended data 
            cConstraint=[cConstraint,diag(G_plus(initL,initL))==1,...
                         diag(G_plus(setunlab,setunlab))<=1,...
                         diag(G_plus(appQind,appQind))<=q,...
                         diag(G_plus(appNQind,appNQind))==0];    
        end
        % When both of the following constraints are added it seems that
        % more reasonable point is queryied(in single point query,Batch
        % mode not tested)
        % cConstraint=[cConstraint,diag(V(setunlab,setunlab))+diag(V(setQuery,setQuery))==ones(n_q,1)];
        % cConstraint=[cConstraint,diag(V(setunlab,setunlab))==diag(V(setQuery,setQuery))-2*q+ones(n_q,1)];
%% Solve Problem        
        sol = optimize(cConstraint,cObjective);
%% Retrieve Result
        if sol.problem==0
            G_data     = value(G_plus);
            g_Dv       = value(g_D(setunlab));
            g_Dall     = value(g_D);
            qinv       = value(G_plus(setQuery,nap+1));
            qresult1   = value(q); % q Value may misguide us, because
            %y_ui may be less than 1 or greater than -1. In these cases,
            %q value may mislead us. 
            % I think the best value is this value which is the nearest
            % value to y_ui*(1-q_i): may be that's because r is not exactly
            % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
            % [-1,1],but we must assume it is {-1,1}
            qyu   =value(G_plus(setunlab,nap+1));% y_u_i * (1-q_i)
            %ra must be equal to absoulte value of qyu but in pratice the
            %value is completely different due to relaxation of y_ui to [-1,1] 
            ra    = value(r);
            if norm1_form == 1
               beta_pv= value(beta_p);
               eta_pv= value(eta_p);
            elseif norm1_form == 2
                beta_pv=value(uDpart);
                eta_pv =value(uQpart);
            end
               
            objpa1= value(c*sum(r));
            objpa2= sum(beta_p)+sum(eta_p(setQuery));
            objt  = value(t);
            cobj  = value(cObjective);
            qresult=1-abs(qyu);
%             [maxq,imaxq]=max(qresult);
%             qbatch=samples_toQuery_from(imaxq);
            
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
            %model_set.g      = 1-g_Dall;
            model_set.h      = 1-g_Dall;
            model_set.noisySamplesInd = [];
        else
            ALresult = 0;
        end
end