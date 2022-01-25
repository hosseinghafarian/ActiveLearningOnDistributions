function [acc_Result, empRisk,Predict,model]...
            = Learnerfortest( isDistData,Options, ModelAndData ,initL,...
                             xTrain,     yTrain,    ...
                             xTest,      yTest ,queryInstance     )
%% Main Learning function
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Variable Initialization 
global qSDP;        %for setting active learning result by SDP Method
global batchSize;   % batchSize for batch active learning
global n_o;
global lnoiseper;
global onoiseper;

initLind = initL>0;
initL    = initL(initLind);

empRisk = 0;
Predict = 0;
% lambda=1;Why initialize when there is a passed parameter for lambda
method        = Options.Querymethod;
classifier    = Options.classifier;
nQueryEachTime= Options.nQueryEachTime;% batch size
transductive  = Options.Transductive;  % Learning and active learning on the same set
lambda        = ModelAndData.lambda;
nL = size(initL,1);
model.dv =0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Calling Train
% First Initialize Classifier Options and varaiables and then call it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sw=1;
if classifier==4  % if diverse subtask active learning
    %% Diverse subTask SDP Active Learning
    %  use method 2 of the same function as classifer 3 
    sw = 2;         % use method 2 
    classifier = 3; % of the same function as classifier= 3
end
%% switch to classifier : TODO: check correctness of the code with accordance with cases 5, 6 and 7
switch classifier % just for cases 1, 2, 3
    case 1 % SVM
            [predict_label, accuracy,decision_values]=...
                libsvm_trainandtest(xTrain(:,initL),yTrain(initL)...
                                   ,xTest,yTest,lambda);
            model.dv = decision_values;
                              
    case 2 % Least Sqaures 
        model.dv =0;
        K = ModelAndData.Kernel(initL',initL');
        lambda = ModelAndData.lambda;
        Klambdainv = inv(K+nL*lambda*eye(nL));        % How can we use A\b instead of inv? it must be done
    case 3 % Convex Relaxation through Semi-definite Programming Using Alternating Minimzation
        lambda = ModelAndData.lambda;
        n = ModelAndData.TrSize;  
        %% Finding labeled, unlabeled and query data indices
        labeled   = initL;
        n_l       = size(labeled,1);
        aln_u     = n - n_l;    % all unlabeled data
        % select only n_u points from unlabled data 
        n_u       = aln_u;              % manually set unlabeled data 
        unlabeleda= setdiff(1:n,labeled)';
        YfChecka  = yTrain(unlabeleda)'; 
        yuforsel  = unlabeleda.* YfChecka;
        ulindexp  = yuforsel(yuforsel>=0)';
        np        = floor(n_u/2);
        if size(ulindexp,2)> np,
            ulindexp = ulindexp(1:np);
        end
        ulindexn  = yuforsel(yuforsel<=0);
        ulindexn  = abs(ulindexn(1:(n_u-np)))';
        unlabeled = sort([ulindexp,ulindexn]');
        YfCheck   = yTrain(unlabeled);
        queryDup  = unlabeled;
        
        if n_u==0 
            return 
        end
        n_q = n_u;
        %% initialize variable q for querying randomly
        q = rand(n_q,1);
        %% Make Kernel matrices of labeled,unlabeled and query data 
        % (data that will be queried from)
        Kll = ModelAndData.Kernel([initL'],[initL']);
        Kuu = ModelAndData.Kernel([unlabeled'],[unlabeled']);
        Klu = ModelAndData.Kernel([initL'],[unlabeled']);
        Kqq = Kuu;
        Klq = Klu;
        Kuq = Kuu;
        
        Yl  = yTrain([initL'])';
        % This code minimizes convex relaxation of transductive svm 
        %[g,Z]= SemiSVMConvexRelax(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
        %% Doing Active minimax algorithm using unlabeled and labeled data
        %[g,Z]= ActiveCRSemiSVM(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
        [q,model_set,Y_set]= ActiveCRSemiSVM03(sw,Kll,Klu,Kuu,Yl,n_l,n_u,lambda);
        model.model = model_set;
        model.type  = 4;               % 4 : diverse subtask
        model.T     = size(model_set); % T : if diverse subtask
        %% Setting Active Learning Results 
        thresholdQ = 0.9;
        [~,ql]=max(q);
        %ql = (q > thresholdQ);
        qSDP = unlabeled(ql);  
        % case 4 is diverse subtask see above     
    case {5,6,7,8,9}
%     case 5 % Convex Relaxation Active Learning
        % In previous cases 1-4, data has moved in order to make Kll,
        % Klu,Kuu, But in these case 5-7, we want to not move data in order
        % to use warm-starting features of Solvers, esp SCS. 
        %% set learning parameters        
        lambda    = ModelAndData.lambda;
        n         = ModelAndData.TrSize; % training data size 
        c         = 1;
        %% Finding labeled, unlabeled and query data indices
        %  the following is based on moving labeled data to 1:n_l and
        %  unlabeled data, if I want to use warmstarting, it must be
        %  changed. 
        labeled   = initL;
        n_l       = size(labeled,1);
        aln_u     = n - n_l;                % all unlabeled data
        % select only n_u points from unlabled data 
        n_u       = aln_u;                  % size of unlabeled data 
        unlabeleda= setdiff(1:n,labeled)';  % unlabeled data indices
        YfChecka  = yTrain(unlabeleda);    % unlabeled data lables 
        yuforsel  = unlabeleda.* YfChecka;  % indices of unlabled data .* labeles of them
        ulindexp  = yuforsel(yuforsel>=0); % indices of positive samples
        ulindexn  = yuforsel(yuforsel<0); % indices of positive samples

        ulindexn  = abs(ulindexn);          % use absolute value of negative instances
        unlabeled = sort([ulindexp;ulindexn]); % sort labeles of da
        YfCheck   = yTrain(unlabeled);
        queryDup  = unlabeled;
        
        if n_u==0 
            return 
        end
        %n_q = n_u;
        %% initialize variable q for querying randomly
        % q = rand(n_q,1);
        
        %% Make Kernel matrices of labeled,unlabeled and query data        
        Yl  = yTrain(initL');
        % This code minimizes convex relaxation of transductive svm 
        %[g,Z]= SemiSVMConvexRelax(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
        %% Convex Relax Active Learning 
        %[g,Z]= ActiveCRSemiSVM(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
        %batchSize = 2;
        %yalmip('solver','scs');
        switch classifier
            case 5
%                  Kll = ModelAndData.Kernel([initL'],[initL']);
%                  Kuu = ModelAndData.Kernel([unlabeled'],[unlabeled']);
%                  Klu = ModelAndData.Kernel([initL'],[unlabeled']);
%                    
%                  [qresult,model_set,Y_set]= ActiveDConvexRelax1(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda);
%                  tq = k_mostlargest(qresult,batchSize);
%                  qi = unlabeled(tq); 
%                  ALresult.queryInd = qi;
                 % The following methods is equivalent and it is tested on some dataset.
                 % In the first method we have beta_p>=0 and eta_p>=0, but in the
                 % second method, u=beta_p-eta_p and we used norm one and
                 % sum of hinge to have the same objective function. 
                 [ALresult,model_set,Y_set]= ActiveDConvexRelax2(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,lambda);   
%                [ALresult,model_set,Y_set]= ActiveDConvexRelax3(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,lambda);
            case 6 % TODO: Change Code to the form of case 5, Change cases 5, 6 to the consider WARMSTART
                % This code minimizes convex relaxation of transductive svm 
                %[g,Z]= SemiSVMConvexRelax(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
                %% Convex Relax Active Learning 
                %[g,Z]= ActiveCRSemiSVM(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)                
                % n_o, it is set globaly:  ;% for 2d sample test
                %  ActiveDConvexRelaxOutlier(sw,Kll,Klu,Kuu,initL,unlabeled,Yl,n_l,n_u,batchSize,lambda)
                [ALresult,model_set,Y_set]= ActiveDConvexRelaxOutlier(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda);
                           
            case 7
                % This code minimizes convex relaxation of transductive svm 
                %[g,Z]= SemiSVMConvexRelax(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
                %% Convex Relax Active Learning 
                %[g,Z]= ActiveCRSemiSVM(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
%                 batchSize = 1;
                %[q,model_set,Y_set]= ActiveDConvexRelaxOutlierEM(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda);
            case 8
                 Kll = ModelAndData.Kernel([initL'],[initL']);
                 Kuu = ModelAndData.Kernel([unlabeled'],[unlabeled']);
                 Klu = ModelAndData.Kernel([initL'],[unlabeled']);
                 
                 [qresult,model_set,Y_set]= ActiveDConvexRelaxOutlierProxADMM2(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda);
                 tq = k_mostlargest(qresult,batchSize);
                 qi = unlabeled(tq); 
                 ALresult.queryInd = qi;
                 % The following methods is equivalent and it is tested on some dataset.
                 % In the first method we have beta_p>=0 and eta_p>=0, but in the
                 % second method, u=beta_p-eta_p and we used norm one and
                 % sum of hinge to have the same objective function. 
%                  [ALresult,model_set,Y_set]= ActiveDConvexRelax2(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,lambda);   
            case 9
                 Kll = ModelAndData.Kernel([initL'],[initL']);
                 Kuu = ModelAndData.Kernel([unlabeled'],[unlabeled']);
                 Klu = ModelAndData.Kernel([initL'],[unlabeled']);
                 n_l = numel(initL);
                 n_u = numel(unlabeled);
                 Ker = ModelAndData.Kernel;
                 ModelAndData.Kernel = [Kll,Klu;Klu',Kuu];
                 initL = 1:n_l;
                 unlabeled = n_l+1:n;
                 lambda_o  = lambda/10;    % assume that lambda_o is lambda/10
                 %[ALresult,model_set,Y_set] = ActiveOutlierSaddle9(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,lambda_o,YfCheck);
                 [ALresult,model_set,Y_set] = ActiveOutlierSaddle8(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,lambda_o,YfCheck);
                 %This is a good version->[ALresult,model_set,Y_set] = ActiveOutlierSaddle7(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,lambda_o,YfCheck);
                 %[ALresult,model_set,Y_set]= ActiveOutlierSaddle6(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,lambda_o,YfCheck);
                 %[ALresult,model_set,Y_set]= ActiveOutlierSaddle5(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,lambda_o,YfCheck);
                 %[ALresult,model_set,Y_set]= ActiveOutlierSaddle4(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,lambda_o,YfCheck);
                 %[ALresult,model_set,Y_set]= ActiveOutlierSaddle3(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,lambda_o,YfCheck);
                 %[ALresult,model_set,Y_set]= ActiveOutlierSaddle1(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,YfCheck);
                 %[ALresult,model_set,Y_set]= ActiveOutlierSEMIInfinite(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda,YfCheck);
                 ModelAndData.Kernel  = Ker;
        end
        yalmip('solver','mosek');
        model.model = model_set;
        model.type  = 4;               % 4 : diverse subtask
        model.T     = size(model_set); % T : if diverse subtask
        %% Setting Active Learning Results 
%         thresholdQ = 0.9;
%         [~,ql]=max(q);
        %ql = (q > thresholdQ);
        qSDP = ALresult.queryInd;             
end

% switch classifier
%     case 6  %if Active Learning for Outlier and label noisy samples
            %%
            % 
            % $$\max_{\alpha} \sum_{i\in D} \alpha_i* g_{D_i}-\frac{2}{\lambda} \alpha^T K\odot G \alpha$$
            % 
if classifier==5 ||classifier==6             
    %% Compute 1-Accuracy with Resulting classifier, 2-Transductive Accuracy, 3- Libsvm, not semisupervised Accuracy
    %% TODO: This part needs carefull consideration, it seems that results are incorrect.
    %        Furthermore, when number of unlabeled data is low, by querying
    %        and therefore reducing number of unlabeled data, accuracy
    %        decreases because denominator decreases and even number of incorrect
    %        classified samples decreases, accuracy decreases or not
    %        increases much. 
    %% TODO : for test data, is test data is outlier based on w_o function?
    %           if it is then it must be said that it is outlier. 
            sdpsettings('solver','mosek');%scs');
%             g_Di = model_set.g;
%             G    = model_set.G;            
           K  = ModelAndData.Kernel;
%             appendDataInd = unlabeled;
%             Kqq = Ke(appendDataInd,appendDataInd);
%             K_q = Ke(:,appendDataInd);                  % Kernel between every thing and setQuery K(labeled and unlabeled,setQuery);
%             K   = [Ke,K_q;K_q',Kqq]; 
%             n_q = size(Kqq,1);
%             nSDP = n+n_q;
%             alpha_p = sdpvar(nSDP,1);
% %             cConstraint= [alpha_p>=0,alpha_p<=1];
%             cConstraint= [alpha_p(1:n)>=0,alpha_p(n+1:nSDP)>=-1,alpha_p<=1];
%             KG    = K.*G;
%             cConstraint= [alpha_p(1:n)>=0,alpha_p(n+1:nSDP)>=-1,alpha_p<=1];
            g_Di = model_set.g(1:n);
            G    = model_set.G(1:n,1:n);
          
            
            alpha_p = sdpvar(n,1);
            cConstraint= [alpha_p>=0,alpha_p<=1];
            K  = ModelAndData.Kernel;
            KG    = K.*G;
            cObjective =-sum(alpha_p.*g_Di)+2/lambda* alpha_p'*KG*alpha_p;
            sol=optimize(cConstraint,cObjective);
            if sol.problem==0
                alpha_pv=value(alpha_p);                             
                sdpsettings('solver','scs');
                setnon_noisy = setdiff(unlabeled,model_set.noisySamplesInd);
                
%                 if classifier==6 
%                     setnon_noisy = setdiff(unlabeled,model_set.noisySamplesInd);
%                     Kerneltest1 = ModelAndData.KA([initL'],:);
%                 else
%                     Kerneltest1 = ModelAndData.KA([initL'],:);
%                 end
                Kerneltest = ModelAndData.KA(1:n,:);% [initL',unlabeled'],:);
                Y_D = zeros(n,1);
                Y_D(initL) = Yl;
                Y_D(unlabeled) = Y_set;
                f_test = Kerneltest'*(alpha_pv.*g_Di.*Y_D);
                y_predtest = sign(f_test);
                if classifier==6 
                    niseq        = bsxfun(@eq, y_predtest, yTest);
                    niseq        = sum(niseq);
                    acc_Result(1)= niseq/size(y_predtest,1);
                else
                    niseq    = bsxfun(@eq, y_predtest, yTest);
                    niseq    = sum(niseq);
                    acc_Result(1) = niseq/size(yTest,1);
                end
            else
                acc_Result(1) = 0;
            end
%%          TODO: labels of semisupervised learning method, is not used for learning with svm, 
%                   1- learning with just labeled samples
%                   2- learning with labeled samples and unlabeled samples using labels predicted by classifier 
%                   
            nolabelnoise_initL = setdiff(initL,model_set.noisySamplesInd);
            if classifier==6 
               [predict_label, accuracy,decision_values]=libsvm_trainandtest(xTrain(:,nolabelnoise_initL),yTrain(nolabelnoise_initL),xTest,yTest,lambda);
            else
               [predict_label, accuracy,decision_values]=libsvm_trainandtest(xTrain(:,nolabelnoise_initL),yTrain(nolabelnoise_initL),xTest,yTest,lambda);
            end
            %[predict_label, accuracy]=libsvm_trainandtest(xTrain(:,initL),yTrain(initL),xTest(:,:),yTest(:),lambda);
            if ~isempty(accuracy)
                acc_Result(3) = accuracy(1);
            else
                acc_Result(3) = -1;
            end
            y_predtest = predict_label;
%%          computing Transductive accuracy  
                % 1- we must not compute outlier/label noisy samples in
                % transductive accuracy
                % 2- may be in non-noisy datasets we must decrease n_o as we move
                % forward
            Yun= sign ( Y_set );
            if classifier==6 
                %Yun = Yun(setnon_noisy);
                YfCheckc = YfCheck;
            else
                YfCheckc = YfCheck;
            end
            niseq = bsxfun(@eq,Yun, YfCheckc);
            niseq = sum(niseq);
            acc_Result(2) = niseq/size(YfCheckc,1);    
            acc_Result    = acc_Result'; 
            return 
end         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calling Test :  TODO: consider returning both transductive and inductive accuracy. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%acc_ML = 0;
switch classifier 
    case 1 
        %% SVM
%         gamma= Options.KOptions.Sigma;
%         K_test = exp(- gamma * pdist2(xTrain(:,model.svind), xTest)) ;
%         f_test = sign(model.alphay(model.svind)' * K_test + model.b) ;
%         niseq = bsxfun(@eq, f_test, yTest);
%         niseq=sum(niseq);
        acc_Result = accuracy(1);    
    case 2 
        %% Least Squares loss function 
        labelX=yTrain([initL]);
        % Computing J(f,lambda) for labeled training data
        empRisk = labelX'*Klambdainv*labelX;
        % Does classifer correctly predicts queryInstance Label?
        KAq=KernelArrayNewInstance(isDistData,xTrain,xTrain(queryInstance),Options.KOptions);
        f=labelX'*Klambdainv*KAq{1}([initL]);
        Sf=sign(f);
        if (Sf==yTrain(queryInstance))
            Predict = true;
        else
            Predict = false;
        end
        Sf=size(yTest,1); 
        niseq=0;
        for it=1:size(xTest)
            KAl=ModelAndData.KA{it}(initL);
            f=labelX'*Klambdainv*KAl;
            Sf(t)=sign(f);
            niseq=niseq+isequal(sign(f),yTest(it));
        end
        acc_Result = niseq/size(xTest,1); 
    case 3 
        %% computing Transductive accuracy 
%             Yun= sign ( Y_set );
%             niseq = bsxfun(@eq,Yun, YfCheck);
%             niseq=sum(niseq);
%             acc_ML = niseq/size(YfCheck,1);
end

end
function [ALresult,model_set,Y_set]= ActiveDConvexRelax1(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda)
        % code of this function before considering warmstart 
        % Aassume all of unlabeled data are candidates for active learning query points    
        % n_l :size(initL,2);
        % n_u :n-n_l;
        % n_q :n_u;
        
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
        % Caution: Don't forget ONED 
        %y_l     =yapp(initL);
        
        
        KVMatrix     = sdpvar(nap+1,nap+1);
        g_D     =sdpvar(nap,1);
        
        % Wow. If you set g_D=1-q,  you get nothing. May be because of the
        % numerical errors
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
function [ALresult,model_set,Y_set]= ActiveDConvexRelax2(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,lambda)
        % code with WARMSTART 
        % Aassume all of unlabeled data are candidates for active learning query points    
        % n_l :size(initL,2);
        % n_u :n-n_l;
        % n_q :n_u;
        norm1_form = 1;
        
        global Warmstart;
        global WMST_beforeStart ;
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
            model_set.g      = 1-g_Dall;
            model_set.noisySamplesInd = [];
        else
            ALresult = 0;
        end
end
function [ALresult,model_set,Y_set]= ActiveDConvexRelaxOutlier(sw,ModelAndData,initL,unlabeled,Yl,n_l,n_u,batchSize,n_o,lnoiseper,onoiseper,lambda)
        %This function is the aggregation of all of the tests, forms of
        %constraints are in constraintform which previously every one of
        %the was a function, now all are in one function for the sake of
        %brevity and integrity. 
        % so far the form that very good statisfies with theory is
        % constraintForm=5, I will use these form for saddleform of the
        % problem( saddle4 or next tests) 
        
        global Warmstart;
        global WMST_beforeStart ;
        global WMST_appendDataInd;
        
        %assert(Warmstart,'This function works only Warmstart is active');
        Warmstart = false;
        Kernel = ModelAndData.Kernel;
        lambda_o  = lambda/10;    % assume that lambda_o is lambda/10
        c   = 5;           % parameter for finding absolute value of y_u.*(1-q) in cost function 
        %% Select Samples to query from : this code not written for part of unlabeled data are being queried from. 
        % Select All of unlabeled samples for Querying
        
        n   = n_l+n_u;     % size of data 
        
        
        %samples_selected_forQuery :samples from unlabeled data to query from
        %setQuery             :indices of query samples appended to kernel
        
        samples_toQuery_from = unlabeled; % Select the set to query from     
        fromunlabtoQuery     = ismember(unlabeled, samples_toQuery_from);  % this variable determines which unlabeled samples are queried
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
        %% Sets of labeled, unlabeled, appended data( a copy of samples to query from) 
        if Warmstart && WMST_beforeStart % if warmstart is true and it's the first time 
            WMST_appendDataInd = samples_toQuery_from; 
            WMST_beforeStart = false;
            appendDataInd = WMST_appendDataInd;
        elseif Warmstart                              % if warmstart is true and it's not the first time then append all of the first time append data 
            appendDataInd = WMST_appendDataInd;
        else                                          % if not warmstart then only append samples to query from 
            appendDataInd = samples_toQuery_from;
        end
        n_a = size(appendDataInd,1);
        nap = n+n_a;                                  % size of appeneded data with Qset // In general Qset can be part of unlabeled data
        n_q = size(samples_toQuery_from,1);
        % in warmstart mode, we may have to appenddata which labeled
        % (because it didn't have labeled at the start and we want to not
        % to change K matrix for warmstart)
        labeled_appenddataind = intersect(appendDataInd,initL);
        isLabeledappend       = false(n,1);
        isLabeledappend(labeled_appenddataind) = true;
        isLabeledappend       = isLabeledappend(appendDataInd); % only retain appended data
        %ylabeled_append       = Yl(labeled_appenddataind);
        setall   = 1:n;
        setunlab = setdiff(setall,initL);
        setQuery = n+1:nap;
        
        %% Make Kernel Matrix for data+appended data
        % appendData is the index of data that a copy of them appended to
        % data for use in active learning 
        Kqq = Kernel(appendDataInd,appendDataInd);
        K_q = Kernel(:,appendDataInd);                  % Kernel between every thing and setQuery K(labeled and unlabeled,setQuery);
        K   = [Kernel,K_q;K_q',Kqq];                    % Kernel appended with queryset Kernels with data and with itself
        %% Select All of unlabeled samples for Querying
        %  setQuery             :indices of appended data (query samples
        %                       appended to kernel)
        
%         Klq = Klu;
%         Kqq = Kuu;
%         Kuq = Kuu; 
%         KD  = [Kll,Klu;Klu',Kuu];    
%         K   = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];    
    
%%      Define YALMIP Variables        
        p     = sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
        w_o   = sdpvar(n,1);          % For w_o function 
        G_plus= sdpvar(nap+1,nap+1);  
        q     = sdpvar(n_q,1);        % Selection variable which we not use it directly
        r     = sdpvar(n_u,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 
        beta_p= sdpvar(nap,1);        % Lagrange for alpha upper bound
        eta_p = sdpvar(nap,1);        % Lagrange for alpha lower bound
        t     = sdpvar(1,1);          %     
        KVMatrix= sdpvar(nap+1,nap+1);
        g_D     = sdpvar(nap,1);
        rl      = sdpvar(n_l,1);
        Pu      = sdpvar(n_u,1); % a variable for Y_u \Phi(X_u)^T w_o
        Yu      = sdpvar(n_u,1);
        ONED    = [ones(n,1);zeros(n_q,1)];
%%      Define Problem, Constraints and Objective  
        constraintForm = 5;
        if constraintForm == 1         % This is My standard method for Direct Convex Outlier Active learning
                KVMatrix   = [K.*G_plus(1:nap,1:nap) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
                cConstraint= [beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
                cConstraint= [cConstraint,sum(q)==batchSize,0<=q,q<=1];% constraints on q 
                % constraints on G_plus       
                cConstraint= [cConstraint,G_plus(nap+1,nap+1)==1,...
                                          G_plus(initL,nap+1)==rl];
                % it is better to substitute p with y_l.*w_o^T\phi(x_i)
                cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
                             diag(G_plus(setunlab,setunlab))<=r,...
                             diag(G_plus(setQuery,setQuery))<=q];
                cConstraint= [cConstraint,diag(G_plus(setunlab,setunlab))<=1-p(setunlab),...
                             diag(G_plus(setunlab,setunlab))<=1-q];

                cConstraint= [cConstraint,G_plus(setQuery,nap+1)==q,...
                             g_D(initL)==zeros(n_l,1),g_D(setQuery)==zeros(n_q,1),g_D(setunlab)<=1-r];
                cConstraint= [cConstraint,-p<=Kernel*w_o<=p,p<=1,rl==Yl-Kernel(initL,:)*w_o];
                % for absolute value of y_u.*(1-pu).*(1-q)
                cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
                cConstraint= [cConstraint,r+q+p(setunlab)==1];
                formp = 1;
                %% Consider percentage of labeled noise data and unlabeled noise data
                if formp==1
                    noiseper=0;%%%%%%%% for test,
                    n_o = 4;%%%%%%%%%%%%
                    c=5;
                    % Form 1: sum(p)<n_o, as a constraint
                    cConstraint=[cConstraint,sum(p(initL))<=n_l*lnoiseper/100,sum(p)<=n_o];
                    %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
                    %cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p(1:n))+norm(1-beta_p(n+1:nap)+eta_p(n+1:nap),1)+c*sum(r);
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r);%+norm(1-r,1);
                elseif formp==2
                    cp         = 2;
                    % Form 2: c*sum(p)+Objective as an addition to objective not as a
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r)+cp*sum(p);
                elseif formp==3
                    c_q     = 1;
                    c_p     = 1;
                    % Form 1: sum(p)<n_o, as a constraint
                    cConstraint=[cConstraint,sum(p)<=n_o];
                    %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+...
                                   c*sum(r)+c_p*norm(p(setunlab),1)+c_q*norm(q,1);
                    %cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);
                end
        elseif constraintForm == 2 % Some changes in the problem formulation above
            %% Changing some constraints above to approximately equivalent form  
                    % changed constraint inequality to equality 
                    %                       diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))<=q
                    %                       to
                    %                       diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))==q
                    % changed constraints g_D(setunlab)<=1-r  to
                    %                          g_D(setunlab)==1-r, and
                    %                          diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))<=r to
                    %                          diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))==r
                %% Constraints
                KVMatrix = [K.*G_plus(1:nap,1:nap) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
                cConstraint=[beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
                cConstraint=[cConstraint,sum(q)==batchSize,0<=q,q<=1];% constraints on q 
                % constraints on G_plus       
                cConstraint=[cConstraint,G_plus(nap+1,nap+1)==1,...
                                         G_plus(initL,nap+1)==rl];
                % it is better to substitute p with y_l.*w_o^T\phi(x_i)
                cConstraint=[cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
                             diag(G_plus(setunlab,setunlab))==r,...% change inquality to equality in this line
                             diag(G_plus(setQuery,setQuery))==q];  % change inquality to equality in this line(last constraint)
                cConstraint=[cConstraint,diag(G_plus(setunlab,setunlab))<=1-p(setunlab),...
                             diag(G_plus(setunlab,setunlab))<=1-q];
                cConstraint=[cConstraint,G_plus(setQuery,nap+1)==q,...
                             g_D(initL)==zeros(n_l,1),g_D(setQuery)==zeros(n_q,1),g_D(setunlab)==1-r];
                cConstraint=[cConstraint,-p<=Kernel*w_o<=p,p<=1,rl==Yl-Kernel(initL,:)*w_o];
                % for absolute value of y_u.*(1-pu).*(1-q)
                cConstraint=[cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
                cConstraint=[cConstraint,r+q+p(setunlab)==1];
                %% Objective function form 
                formp = 1;
                if formp==1
                    % Form 1: sum(p)<n_o, as a constraint
                    cConstraint=[cConstraint,sum(p)<=n_o];
                    %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r);        
                else
                    cp       =2;
                   % Form 2: c*sum(p)+Objective as an addition to objective not as a
                   cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r)+cp*sum(p); %addition of sum(q) didn't help+3*sum(q);
                end
        elseif constraintForm == 3 % 
                %% Change : Substitute q with  G_plus(n+1:nap,nap+1)
                                %       q is G_plus(n+1:nap,nap+1)
                KVMatrix = [K.*G_plus(1:nap,1:nap) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
                cConstraint=[beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
                cConstraint=[cConstraint,sum(G_plus(setQuery,nap+1))==batchSize,0<=G_plus(setQuery,nap+1),G_plus(setQuery,nap+1)<=1];% constraints on q 
                % constraints on G_plus       
                cConstraint=[cConstraint,G_plus(nap+1,nap+1)==1,...
                                         G_plus(initL,nap+1)==rl];
                % it is better to substitute p with y_l.*w_o^T\phi(x_i)
                cConstraint=[cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
                             diag(G_plus(setunlab,setunlab))==r,...% change inquality to equality in this line
                             diag(G_plus(setQuery,setQuery))==G_plus(setQuery,nap+1)];  % change inquality to equality in this line(last constraint)
                cConstraint=[cConstraint,diag(G_plus(setQuery,setQuery))<=1-p(setunlab),...
                             diag(G_plus(setunlab,setunlab))<=1-G_plus(setQuery,nap+1)];

                cConstraint=[cConstraint,g_D(initL)==zeros(n_l,1),g_D(setQuery)==zeros(n_q,1),g_D(setunlab)==1-r];
                cConstraint=[cConstraint,r+G_plus(setQuery,nap+1)+p(setunlab)==1];       


                cConstraint=[cConstraint,-p<=Kernel*w_o<=p,p<=1,rl==Yl-Kernel(initL,:)*w_o];
                % for absolute value of y_u.*(1-pu).*(1-q)
                cConstraint=[cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
                formp = 1;
                if formp==1
                    % Form 1: sum(p)<n_o, as a constraint
                    cConstraint=[cConstraint,sum(p)<=n_o];
                    %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r);
                else
                     cp       =2;
                     % Form 2: c*sum(p)+Objective as an addition to objective not as a
                     cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r)+cp*sum(p); %addition of sum(q) didn't help+3*sum(q);
                end
        elseif constraintForm ==5
                KVMatrix   = [K.*G_plus(1:nap,1:nap) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
                cConstraint= [beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
                                            %ok
                cConstraint= [cConstraint,sum(q)==batchSize,...
                    0<=q,q<=1];% constraints on q%%implict 
                % constraints on G_plus       ok
                cConstraint= [cConstraint,G_plus(nap+1,nap+1)==1,...
                                          G_plus(initL,nap+1)==rl];
                % it is better to substitute p with y_l.*w_o^T\phi(x_i)
                cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
                             diag(G_plus(setunlab,setunlab))==r,...
                             diag(G_plus(setQuery,setQuery))==q];
                %cConstraint= [cConstraint,diag(G_plus(setunlab,setunlab))<=1-p(setunlab),...
                %            diag(G_plus(setunlab,setunlab))<=1-q];

                cConstraint= [cConstraint,G_plus(setQuery,nap+1)==q,...
                             g_D(initL)==p(initL),g_D(setQuery)==zeros(n_q,1),g_D(setunlab)==1-r];
                cConstraint= [cConstraint,-p<=Kernel*w_o<=p,p<=1,rl==Yl-Kernel(initL,:)*w_o];
                % for absolute value of y_u.*(1-pu).*(1-q)
                cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
                cConstraint= [cConstraint,r+q+p(setunlab)==1];
                formp = 1;
                %% Consider percentage of labeled noise data and unlabeled noise data
                if formp==1
                    noiseper=0;%%%%%%%% for test,
                    n_o = 4;%%%%%%%%%%%%
                    c=5;
                    % Form 1: sum(p)<n_o, as a constraint
                    cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
                    %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
                    %cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p(1:n))+norm(1-beta_p(n+1:nap)+eta_p(n+1:nap),1)+c*sum(r);
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r);%+norm(1-r,1);
                elseif formp==2
                    cp         = 2;
                    % Form 2: c*sum(p)+Objective as an addition to objective not as a
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+c*sum(r)+cp*sum(p);
                elseif formp==3
                    c_q     = 1;
                    c_p     = 1;
                    % Form 1: sum(p)<n_o, as a constraint
                    cConstraint=[cConstraint,sum(p)<=n_o];
                    %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
                    cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(setQuery))+...
                                   c*sum(r)+c_p*norm(p(setunlab),1)+c_q*norm(q,1);
                    %cObjective = t+lambda_o*w_o'*Kernel*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);
                end
        end
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            inpsolverobjective = value(cObjective);
            G_datap = value(G_plus);
            if constraintForm == 3
                qv   = value(G_plus(setQuery,nap+1));
            else
                qv     = value(q); % q Value may misguide us, because it won't consider y_ui
            end
            pv     = value(p);
            g_Dv   = value(g_D(setunlab));
            g_Da   = value(g_D);
            qinv   = value(G_plus(setQuery,nap+1));
           
            %Puv    = value(Pu);
            w_ov    = value(w_o);
            %Yuv    = value(Yu);
            sv     = value(g_D);
            rlv    = value(rl);
            beta_pv= value(beta_p);
            eta_pv = value(eta_p);
            rv     = value(r);
            G_plusv= value(G_plus);
            G_data = value(G_plus(1:n,1:n));
            %% Discussion about using use of qyu for obtaining q
            %y_ui may be less than 1 or greater than -1. In these cases,
            %q value may mislead us. 
            % I think the best value is this value which is the nearest
            % value to y_ui*(1-q_i): may be that's because r is not exactly
            % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
            % [-1,1],but we must assume it is {-1,1}
            
            %ra must be equal to absoulte value of qyu but in pratice the
            %value is completely different due to relaxation of y_ui to [-1,1] 
            %sa    =value(s);
%             if formp==1
%                  qresult=1-abs(qyu);
%                  qresult=qresult.*(1-pv(n_l+1:n));
%             else
%                  qresult=qv;    
%             end
            %% Obtaining results
            qyu   =value(G_plus(setunlab,nap+1));% y_u_i * (1-q_i)    
            alsubtype = 4;
            % find largest p, (they are the noisiest)
            noisysamplesInd = k_mostlargest(pv(1:n),n_o);
            % retain only unlabeled samples
            isunlabelednoisy  = ismember(noisysamplesInd,unlabeled);
            noisysamplesInd   = noisysamplesInd(isunlabelednoisy);
            % change indices of noisy samples to indices for unlabeled
            % samples by subtracting n_l: this is wrong if we didn't move
            % labeled samples to the begining 
            %noisysamplesInd   = noisysamplesInd-n_l;
            qresult =zeros(n,1);
            %noisysamplesqfrom = fromunlabtoQuery(noisysamplesInd);   % noisy samples which are also queried. 
            
            if alsubtype==1
                qresult(samples_toQuery_from) = qv;
            elseif alsubtype==2
                qresult(samples_toQuery_from) = qv;
                qresult = qresult.*(1-pv);
            elseif alsubtype==3
                qresult(samples_toQuery_from) = 1-abs(qyu);
                qresult = qresult.*(1-pv);
                
            elseif alsubtype==4
                qresult(samples_toQuery_from) = 1-abs(qyu);
                qresult = qresult.*(1-pv);
                
                qresult(noisysamplesInd) = 0;          % TODO: it must be checked that p_i for noisydata are significantly larger than others,
                                                       % otherwise if for example all of them are zero it has no meaning
            elseif alsubtype==5
                qresult(samples_toQuery_from) = qv;
                qresult = qv.*(1-pv);    
                
                qresult(noisysamplesInd) = 0; 
            elseif alsubtype==6
                qresult(samples_toQuery_from) = qv;    
                qresult(noisysamplesInd) = 0;
            end
            tq = k_mostlargest(qresult,batchSize);
            qi = tq;%unlabeled(tq); 


            Y_set  = sign(value(qyu));
            % set learning parameters 
            model_set.beta_p = beta_pv;
            model_set.eta_p  = eta_pv;
            model_set.w_oxT_i= w_ov'*Kernel;
            model_set.G      = G_data;
            model_set.g      = value(1-g_D);
            model_set.noisySamplesInd = noisysamplesInd;
            
            ALresult.q = zeros(n,1);
            ALresult.q = qresult;
            ALresult.samples_toQuery_from = samples_toQuery_from;
            tq = k_mostlargest(qresult,batchSize);
            ALresult.queryInd = tq;% samples_toQuery_from(tq);
            ALresult.qBatch = zeros(n_u,1);
            ALresult.qBatch(ALresult.queryInd) = 1;     
            %% This done for Saddle point problem
            alphas = sdpvar(nap,1);
            lo     = [zeros(n,1);-ones(nap-n,1)];
            up     = ones(nap,1);
            % r_D=1_D-pv-qv, G,alphatild = alpha(1:n)
            
            r_D = ones(n,1)-pv;
            r_D(unlabeled) = r_D(unlabeled)-qv;
            r_D = [r_D;zeros(nap-n,1)];
            iseq = r_D==ONED-g_Da
            Ga = value(G_plus(1:nap,1:nap));
            KG2  =K.*Ga;
            cObjective  = -r_D'*alphas+1/(2*lambda)*alphas'*KG2*alphas;
            cConstraint = [alphas>=lo,alphas<=up];
            sol = optimize(cConstraint,cObjective);
            if sol.problem==0
                alphav = value(alphas);
                save('datafile.mat','alphav','G_plusv','pv','qv','inpsolverobjective');
            end
            %check alphav against value of betav etav, 
            rhs = lambda*(ONED-g_Da+eta_pv-beta_pv);
            lhs = (K.*G_plusv(1:nap,1:nap))*alphav;
            difl=lhs-rhs;
            alphav2   = lambda*pinv(K.*G_plusv(1:nap,1:nap))*rhs;
            pqv       = pv(setunlab) + qv;
            saddleobj = sum(alphav(1:n))-alphav(1:n)'*[p(1:2);pqv]-1/(2*lambda)*alphav'*(K.*Ga)*alphav+c*(n-2)-c*ones(n-2,1)'*pqv+lambda_o*w_ov'*Kernel*w_ov/2;
            norm(difl)
        end
end
function S=projSDP(M)
[V,D]=eig(M);
D_p  =diag(D);
D_p  =bsxfun(@max,D_p,0);
S    =V'*diag(D_p)*V;
end
% deprecated functions
function [qresult,model_set,Y_set]= ActiveDConvexRelaxOutlier5(sw,Kll,Klu,Kuu,initL,unlabeled,Yl,n_l,n_u,batchSize,lambda)
        % This function is deprecated 
        % Exactly the same as  ActiveDConvexRelaxOutlier, but with abs(r)+u=1-p, and u equals convex relax of q.*(1-p), so 
        % u<= q, u<= 1-p, But unfortunately the results was approximatly same, not better.
        % Aassume all of unlabeled data are candidates for active learning query points    
        % n_l :size(initL,2);
        % n_u :n-n_l;
        % n_q :n_u;
       
        lambda_o = lambda/10;
       %% Select All of unlabeled samples for Querying
        n_q = n_u;
        n   = n_l+n_u;     % size of data 
        nap = n_l+n_u+n_q; % size of appeneded data with Qset // In general Qset can be part of unlabeled data
        n_o      = 4;   
        
        samples_toQuery_from = unlabeled; % Select the set to query from 
        %samples_selected_forQuery :samples from unlabeled data to query from
        %setQuery             :indices of query samples appended to kernel
        
        Klq = Klu;
        Kqq = Kuu;
        Kuq = Kuu; 
        KD  = [Kll,Klu;Klu',Kuu];    
        K   = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];    
        c   = 5;     %parameter for finding absolute value of y_u.*(1-q) in cost function 
        n   = n_l+n_u;
        nap   =n_l+n_u+n_q;
        
        p     =sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
        w_o   =sdpvar(n,1);          % For w_o function 
        G_plus     =sdpvar(nap+1,nap+1);  
        q     =sdpvar(n_u,1);        % Selection variable which we not use it directly
        r     =sdpvar(n_u,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 
        beta_p=sdpvar(nap,1);        % Lagrange for alpha upper bound
        eta_p =sdpvar(nap,1);        % Lagrange for alpha lower bound
        t     =sdpvar(1,1);          % 
        u     = sdpvar(n_q,1);
        % Unlabeled data is from 1 to n_l
        initL =[1:n_l]; 
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        setQuery =n+1:nap;
        ONED     = [ones(n,1);zeros(n_q,1)];
        %y_l     =yapp(initL);
        
        
        KVMatrix     = sdpvar(nap+1,nap+1);
        g_D     =sdpvar(nap,1);
        rl    =sdpvar(n_l,1);
        Pu    =sdpvar(n_u,1); % a variable for Y_u \Phi(X_u)^T w_o
        Yu    =sdpvar(n_u,1);
        KVMatrix = [K.*G_plus(1:nap,1:nap) ,ONED-g_D+eta_p-beta_p;(ONED-g_D+eta_p-beta_p)',2*t/lambda];
        
        
        cConstraint=[beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; % nonnegativity constraints
        cConstraint=[cConstraint,sum(q)==batchSize,0<=q,q<=1];  % constraints on q 
        cConstraint=[cConstraint,0<=u,u<=q,u<=1-p(n_l+1:n),sum(u)==batchSize];    % constraints on u 
        % constraints on G_plus       
        cConstraint=[cConstraint,G_plus(nap+1,nap+1)==1,...
                                 G_plus(initL,nap+1)==rl];
        % it is better to substitute p with y_l.*w_o^T\phi(x_i)
                              % Constraints about G_ll and G_QQ
        cConstraint=[cConstraint, diag(G_plus(1:n_l,1:n_l))==1-p(1:n_l)       ,...
                                  diag(G_plus(n_l+n_u+1:nap,n+1:nap))<=q,...
                                  diag(G_plus(n_l+n_u+1:nap,n+1:nap))<=u];
                              % Constraints about G_uu
        cConstraint=[cConstraint, diag(G_plus(n_l+1:n,n_l+1:n))<=1-p(n_l+1:n),...
                                  diag(G_plus(n_l+1:n,n_l+1:n))<=r,...  
                                  diag(G_plus(n_l+1:n,n_l+1:n))<=1-q];
                 
        cConstraint=[cConstraint,G_plus(n+1:nap,nap+1)==u,... % change G_plus(n+1:nap,nap+1)==q to G_plus(n+1:nap,nap+1)==u
                     g_D(initL)==zeros(n_l,1),g_D(n+1:nap)==zeros(n_q,1),g_D(setunlab)<=1-r];
                 
        
                 
        cConstraint=[cConstraint,-p<=KD*w_o<=p,p<=1,rl==Yl-KD(1:n_l,:)*w_o];
        % for absolute value of y_u.*(1-pu).*(1-q)
        cConstraint=[cConstraint,r>=G_plus(n_l+1:n,nap+1),r>=-G_plus(n_l+1:n,nap+1)];
        cConstraint=[cConstraint,r+u+p(n_l+1:n)==1];%%change r+q+p(n_l+1:n)==1 to r+u+p(n_l+1:n)==1
        
        
        formp = 1;
        if formp==1
            c=1;
              
            % Form 1: sum(p)<n_o, as a constraint
            cConstraint=[cConstraint,sum(p)<=n_o];
            %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
            %cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p(1:n))+norm(1-beta_p(n+1:nap)+eta_p(n+1:nap),1)+c*sum(r);
            cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);%+norm(1-r,1);
        elseif formp==2
             n_o      = 4;
             cp       =2;
            % Form 2: c*sum(p)+Objective as an addition to objective not as a
            cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r)+cp*sum(p);
        elseif formp==3
            n_o      = 4;    
            c_q     = 1;
            c_p     = 1;
            % Form 1: sum(p)<n_o, as a constraint
            cConstraint=[cConstraint,sum(p)<=n_o];
            %cConstraint=[cConstraint,sum(g_D)+Yl.*rl>=n-batchSize-n_o];
            cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+...
                           c*sum(r)+c_p*norm(p(n_l+1:n),1)+c_q*norm(q,1);
            %cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);
        end
       
%%                            
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))+diag(V(setQuery,setQuery))==ones(n_q,1)];
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))==diag(V(setQuery,setQuery))-2*q(setunlab)+ones(n_q,1)];
        
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            Vprimal     = value(G_plus);
            qv     = value(q); % q Value may misguide us, because
            pv     = value(p);
            g_Dv   = value(g_D(n_l+1:n));
            qinv   = value(G_plus(n_l+n_u+1:nap,nap+1));
           
            %Puv    = value(Pu);
            w_ov    = value(w_o);
            %Yuv    = value(Yu);
            sv     = value(g_D);
            rlv    = value(rl);
            beta_pv= value(beta_p);
            eta_pv = value(eta_p);
            rv     = value(r);
            G_data = value(G_plus(1:n,1:n));
            
            %y_ui may be less than 1 or greater than -1. In these cases,
            %q value may mislead us. 
            % I think the best value is this value which is the nearest
            % value to y_ui*(1-q_i): may be that's because r is not exactly
            % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
            % [-1,1],but we must assume it is {-1,1}
            qyu   =value(G_plus(n_l+1:n,nap+1));% y_u_i * (1-q_i)
            %ra must be equal to absoulte value of qyu but in pratice the
            %value is completely different due to relaxation of y_ui to [-1,1] 
            %sa    =value(s);
%             if formp==1
%                  qresult=1-abs(qyu);
%                  qresult=qresult.*(1-pv(n_l+1:n));
%             else
%                  qresult=qv;    
%             end
            alsubtype = 4;
            if alsubtype==1
                qresult = qv;
            elseif alsubtype==2
                qresult = qv.*(1-pv(n_l+1:n));
            elseif alsubtype==3
                qresult=1-abs(qyu);
                qresult=qresult.*(1-pv(n_l+1:n));
            elseif alsubtype==4
                qresult=1-abs(qyu);
                qresult=qresult.*(1-pv(n_l+1:n));
                noisysamplesInd = k_mostlargest(pv(n_l+1:n),n_o);
                qresult(noisysamplesInd) = 0;          % TODO: it must be checked that p_i for noisydata are significantly larger than others,
                                                       % otherwise if for example all of them are zero it has no meaning
            elseif alsubtype==5
                qresult = qv.*(1-pv(n_l+1:n));    
                noisysamplesInd = k_mostlargest(pv(n_l+1:n),n_o);
                qresult(noisysamplesInd) = 0; 
            elseif alsubtype==6
                qresult = qv;    
                noisysamplesInd = k_mostlargest(pv(n_l+1:n),n_o);
                qresult(noisysamplesInd) = 0;
            end
            tq = k_mostlargest(qresult,batchSize);
            qi = unlabeled(tq); 

            Y_set  = sign(value(qyu));
            % set learning parameters 
            model_set.beta_p = beta_pv;
            model_set.eta_p  = eta_pv;
            model_set.w_oxT_i= w_ov'*KD;
            model_set.G      = G_data;
            model_set.g      = value(1-g_D);
            model_set.noisySamplesInd = noisysamplesInd;
            
            ALresult.q                       = zeros(n,1);
            ALresult.q(samples_toQuery_from) = qresult;
            ALresult.samples_toQuery_from    = samples_toQuery_from;
            tq = k_mostlargest(qresult,batchSize);
            ALresult.queryInd = samples_toQuery_from(tq);
            ALresult.qBatch   = zeros(n_u,1);
            ALresult.qBatch(ALresult.queryInd) = 1;
            
        end
end
function [qresult,model_set,Y_set]= ActiveDConvexRelaxOutlierEM(sw,Kll,Klu,Kuu,Yl,n_l,n_u,batchSize,lambda)
        % Aassume all of unlabeled data are candidates for active learning query points    
        % n_l :size(initL,2);
        % n_u :n-n_l;
        % n_q :n_u;
       
        lambda_o = lambda/10;
       
        
        
        n_q = n_u;
        Klq = Klu;
        Kqq = Kuu;
        Kuq = Kuu; 
        KD  = [Kll,Klu;Klu',Kuu];    
        K   = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];    
        c   = 5;     %parameter for finding absolute value of y_u.*(1-q) in cost function 
        n   = n_l+n_u;
        nap   =n_l+n_u+n_q;
        
        p     =sdpvar(n,1);          % For absolute value of Outlier function w_o^T\phi(x_i)
        w_o   =sdpvar(n,1);          % For w_o function 
        G_plus     =sdpvar(nap+1,nap+1);  
        q     =sdpvar(n_u,1);        % Selection variable which we not use it directly
        r     =sdpvar(n_u,1);        % For Filtering out y_ui : absolute value of y_ui (1-y_ui w_o(\phi(x_i)) 
        beta_p=sdpvar(nap,1);        % Lagrange for alpha upper bound
        eta_p =sdpvar(nap,1);        % Lagrange for alpha lower bound
        t     =sdpvar(1,1);          % 
        % Unlabeled data is from 1 to n_l
        initL =[1:n_l]; 
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        setQuery =n+1:nap;
    
        KVMatrix     = sdpvar(nap+1,nap+1);
        g_D     =sdpvar(nap,1);
        rl    =sdpvar(n_l,1);
        Pu    =sdpvar(n_u,1); % a variable for Y_u \Phi(X_u)^T w_o
        Yu    =sdpvar(n_u,1);
        KVMatrix = [K.*G_plus(1:nap,1:nap) ,1-g_D+eta_p-beta_p;(1-g_D+eta_p-beta_p)',2*t/lambda];
        
        
        cConstraint=[beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0];
        cConstraint=[cConstraint,sum(q)==batchSize,0<=q,q<=1];

        cConstraint=[cConstraint,G_plus(nap+1,nap+1)==1,...
                                 diag(G_plus(1:n_l+n_u,1:n_l+n_u))<=1-p,G_plus(initL,nap+1)==rl];
        cConstraint=[cConstraint,r>=G_plus(n_l+1:n,nap+1),r>=-G_plus(n_l+1:n,nap+1),...
                                r+q+p(n_l+1:n)==1]; 
                                %r+q==1-Pu,Pu<=1-q];
        cConstraint=[cConstraint,G_plus(n+1:nap,nap+1)==q,...
                     g_D(initL)==zeros(n_l,1),g_D(n+1:nap)==zeros(n_q,1),g_D(setunlab)==q];
 %       cConstraint=[cConstraint,sum(diag(V(n_l+1:n,n_l+1:n)))>= n_u-batchSize-n_o];         
        formp = 1;
        if formp==1
         n_o      = 4;    
        % Form 1: sum(p)<n_o, as a constraint
        cConstraint=[cConstraint,-p<=KD*w_o<=p,p<=1,sum(p)<=n_o,rl==Yl-KD(1:n_l,:)*w_o];
        cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);
        else
             n_o      = 4;
            cp       =2;
        % Form 2: c*sum(p)+Objective as an addition to objective not as a
        cConstraint=[cConstraint,-p<=KD*w_o<=p,p<=1,rl==Yl-KD(1:n_l,:)*w_o];
        cObjective = t+lambda_o*w_o'*KD*w_o/2+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r)+cp*sum(p);
        end
             
        % The following constraints which is an essential part of the
        % relaxation proposed is very good even more than the next two line
        % constraints.
        if formp==1
        cConstraint=[cConstraint,diag(G_plus(1:n_l,1:n_l))==1-p(1:n_l),...
                     diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))<=r,...
                     diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))<=q];
        else
        cConstraint=[cConstraint,diag(G_plus(1:n_l,1:n_l))==1-p(1:n_l),...
                     diag(G_plus(n_l+1:n_l+n_u,n_l+1:n_l+n_u))<=1,...
                     diag(G_plus(n_l+n_u+1:nap,n_l+n_u+1:nap))<=q];
        end
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))+diag(V(setQuery,setQuery))==ones(n_q,1)];
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))==diag(V(setQuery,setQuery))-2*q(setunlab)+ones(n_q,1)];
        
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            Vprimal     = value(G_plus);
            qv     = value(q); % q Value may misguide us, because
            pv     = value(p);
            g_Dv        = value(g_D(n_l+1:n));
            qinv        = value(G_plus(n_l+n_u+1:nap,nap+1));
            qresult1     = value(q);
            %Puv    = value(Pu);
            w_ov   = value(w_o);
            %Yuv    = value(Yu);
            sv     = value(g_D);
            rlv    = value(rl);
            beta_pv= value(beta_p);
            rv     = value(r);
            
            
            %y_ui may be less than 1 or greater than -1. In these cases,
            %q value may mislead us. 
            % I think the best value is this value which is the nearest
            % value to y_ui*(1-q_i): may be that's because r is not exactly
            % the absolute value of V(n_l+1:n,nap+1) and y_ui is relaxed to
            % [-1,1],but we must assume it is {-1,1}
            qyu   =value(G_plus(n_l+1:n,nap+1));% y_u_i * (1-q_i)
            %ra must be equal to absoulte value of qyu but in pratice the
            %value is completely different due to relaxation of y_ui to [-1,1] 
            %sa    =value(s);
            tv(:,1)=qv;
            tv(:,2)=pv(n_l+1:n_l+n_u);
            tv(:,3)=rv;
            tv(:,4)=qyu;
            if formp==1
            qresult=1-abs(qyu);
            else
            qresult=qv;    
            end
            [maxq,imaxq]=max(qresult);
            qbatch=imaxq;
            Y_set  = sign(value(qyu));
            model_set =1;% for now it is set 1 to figure out it?
        end
end
