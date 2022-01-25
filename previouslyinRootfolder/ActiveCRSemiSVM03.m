function  [ALresult,model_set,Y_set]= ActiveCRSemiSVM03(WMST_beforeStart, Model, learningparams)
global cnstData
% This function and activediversemultitask have exactly the same code. But
% there is two copies of them. I must delete unrelevant code from each of
% them. 
ALresult.active  = true;
sw        = 1;%%Doing Active minimax algorithm using unlabeled and labeled data
initL     = cnstData.initL;
unlabeled = cnstData.unlabeled;
n_l       = cnstData.n_l;
n_u       = cnstData.n_u;
batchSize = cnstData.batchSize;
n_o       = cnstData.n_o;
lambda    = learningparams.lambda;
Yl        = cnstData.Yl;
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
Kll = ModelInfo.Kernel([initL'],[initL']);
Kuu = ModelInfo.Kernel([unlabeled'],[unlabeled']);
Klu = ModelInfo.Kernel([initL'],[unlabeled']);
Kqq = Kuu;
Klq = Klu;
Kuq = Kuu;
switch sw
    case 1 % Convex Relax Semi-supervised SVM 
        [q,A]= ActiveConvexRelaxSemiSVM(Kll,Klu,Kuu,Yl,n_l,n_u,lambda);
    case 2 % Diverse subTask 
        T   = 1;% number of sub tasks 
        trainOptions.lambda=lambda;
        trainOptions.muPar =0.1;
        %% select subtasks data
        subTaskdata = selectsubtaskData(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,T);
        %% train subtasks
        [q,model_set,Y_set]= diverseSubTaskAL(subTaskdata,trainOptions);
end
ALresult.q = q;
end
function  [q,A]= ActiveConvexRelaxSemiSVM(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
% This function minimzes the SDP Objective function of Convex Relaxation of
% Semisupervised SVM using Alternating minimization
% It is changed at 93/11/25 in order to use method of paper: "A hybrid
% algorithm for convex semidefinite optimization"
%% comment:Finding optimal with search to compare result 
%    [qi,OPTMAT,ZA]=findOptimalWithSearch(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)

%% Initializing Q: the subset of unlabeled data to query from 
    % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
    % of using only a part of unlabeled data searching for query points
    n_q = n_u; % consider all of unlabeled data for selecting query
    bSize  = 2;
    g = 0; % For now! 
%% Initialize some constants    
    ONEA = ones(n_l+n_u+n_q,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);
    ONEQMATRIX = ONEQ*ONEQ';
%% Prepare Kernel Matrix 
    Klq = Klu;
    Kqq = Kuu;
    Kuq = Kuu;
%% Initalize starting point of (query,Lables)for Alternating Minimization      
    q   = bSize*ones(n_u,1)/n_u;
    % initialize label for unlabled data randomly
    % use it for making label matrix A 
    Yu =sign(-1+2*rand(n_u,1));
    Ya = [Yl;Yu;ONEQ];
    AMatrix = Ya*Ya';
%% Starting Alternating minimization Loop for query and (Learning,Lables)    
    for i=1:4 
%% comment: Some previous code to Optimizing Directly beta,eta and A() 
        % qc = q;    
        % goto function: OptimizeR_beta_eta
%% Fix q and ADMM Optimizing Semi-supervised SVM(Learning,Lables)
        qc = q;
        acceptableError = 2;% for controling error if necessary
        [AMatrix,beta_Var,eta_Var]=ADMMOpt_SemiSVM03(acceptableError,Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,AMatrix);
%% Fix (Learning,Lables) and Optimizing to find q: query point(s)
        % the following function will do the same work
        %[q,A]=UpdateQuery(n_l,n_u,Kll,Klu,Kuu,eta_Varc,beta_Varc);
        eta_Varc=eta_Var;
        beta_Varc=beta_Var;
  
        cvx_begin sdp
            variable   A_uu(n_u,n_u)% lable matrix for unlabeled data
            variable   A_uq(n_u,n_q)% lable matrix for unlabeled and query data
            variable   A_lu(n_l,n_u)% lable matrix for labeled and query data
            variable   q(n_q,1)     % query variable, the largest elements are for querying
            variable   tVar         % objective function
            expression K            % Kernel Matrix as a whole 
            expression A            % Lable Matrix as a whole  
            expression ASmall1      
            expression ASmall2
            expression r_q          
            expression p_q
            
            r_q=[ONEL;1-q;q];
            p_q=[zeros(n_l+n_u,1);q];
            K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
            % This is the lable matrix which we want to estimate it's
            % unlabeled part
            A = [Yl*Yl',A_lu,Yl*ONEQ';A_lu',A_uu,A_uq;ONEQ*Yl',A_uq',ONEQMATRIX];
            ASmall1 = [Yl*Yl',A_lu;A_lu',A_uu];
            ASmall2 = [A_uu,A_uq';A_uq,ONEQMATRIX];
            % Declare Optimization problem for CVX
            minimize (tVar)
            subject to
               0 <= q <=1;
               %A>=0;
               ASmall1>=0;
               ASmall2>=0;
               q'*ONEQ==bSize;
               diag(A_uu)==ONEU;
               [K .* A,ONED-beta_Varc+eta_Varc;(ONED-beta_Varc+eta_Varc)',2*(tVar-beta_Varc'*r_q-eta_Varc'*p_q)/lambda ]>=0;
        cvx_end  
    end
end

function [AMatrix,beta_Var,eta_Var]=ADMMOpt_SemiSVM02(acceptableError,Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,AMatrix)
    % This function attempts to optimizes Semi-supervised SVM using AM(
    % Alteranting Minimzation). But unfortunately, the dual derived
    % doesn't work with this problem. There is a one-dimensional variable rho which we
    % attempt to find it's optimum by changing it's value in a range.
    % The Dual and Primal problem doesn't have the same objective function.
    % I post this problem in stackexchange.com and cvx forum. 
    % ADMM_Label032 works correctly but it is very slow. The aim was to
    % improve it's speed. 
    Error = acceptableError +10 ;
    [eta_Var,beta_Var] = ADMM_Classifier(Kll,Klu,Kuu,n_l,n_u,lambda,q,AMatrix);
    Apre = AMatrix;
    [AMatrix1,optval1,dualVars] = ADMM_Label032(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var); 
    dualVars.rho = dualVars.drho; 
    stepsize = 0.0001;
    i = 1;
    while (dualVars.rho > 0 )
        [AMatrix2,optval2] = ADMM_Label041(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var,dualVars);
        nne =  norm(AMatrix2-AMatrix1,'fro')/norm(AMatrix1,'fro')
        RHODATA(i,1)=dualVars.rho;
        RHODATA(i,2)=nne;
        RHODATA(i,3)=optval2;
        RHODATA(i,4)=optval1;
        dualVars.rho = dualVars.rho - stepsize;
        i= i +1;
        if ( mod(i,50)==0 )
            save('rhodatafile','RHODATA');
        end
        if ( mod(i,120)==0 )
            save('rhodatafile','RHODATA');
        end
    end
    save('rhodatafile','RHODATA');
%%     [AMatrix2] = ADMM_Label04(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var,AMatrix1);
%     Error12 = norm(AMatrix1-AMatrix2,'fro')/norm(AMatrix1,'fro')
%     h = 800;
%     while Error > acceptableError 
%          
%         [AMatrix3] = ADMM_Label06(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var,AMatrix1);   
%         %[AMatrix4] = ADMM_Label07(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var); 
%         Error12 = norm(AMatrix3-AMatrix2,'fro')/norm(AMatrix1,'fro')
%         Error12 = norm(AMatrix3-AMatrix1,'fro')/norm(AMatrix1,'fro')
%         h=h+10;
% %         Error23 = norm(AMatrix3-AMatrix2,'fro');
% %         Error34 = norm(AMatrix3-AMatrix4,'fro');
%     end
end
function [AMatrix,beta_Var,eta_Var]=ADMMOpt_SemiSVM03(acceptableError,Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,AMatrix)
%% Optimizing to compute Learning and Lables using Alternating Minimzation
    Error = acceptableError +10 ;
    for i=1:4
        %% Fix A ( Label Matrix ) and Optimizes beta,eta
        [eta_Var,beta_Var] = ADMM_Classifier(Kll,Klu,Kuu,n_l,n_u,lambda,q,AMatrix);
        Apre = AMatrix;
        %% Fix beta,eta and Optimize A( Label Matrix )
        % ADMM_Label032 works correctly, but it is very slow, to improve
        % speed, I wrote ADMM_LabelSDPConstrReduction. 
        % Unfortunately, I don't know how to compute primal variable from dual.
        % It is needed to check it more, but I think it is correct. 
        % I must find a way to compute the primal variables. 
        [AMatrix1,optval1,dualVars] = ADMM_Label032(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var); 
        dualVars.rho = dualVars.drho;
        
        [AMatrix2,optval2] = ADMM_LabelSDPConstrReduction(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var,AMatrix1);
        nne =  norm(AMatrix2-AMatrix1,'fro')/norm(AMatrix1,'fro')
        AMatrix=AMatrix1;
    end
end
%% This function fixes A ( Lable Matrix ) and Optimizes beta and eta
function [eta_Varc,beta_Varc] = ADMM_Classifier(Kll,Klu,Kuu,n_l,n_u,lambda,q,AMatrix)
        bSize  = 2;
        g = 0; % For now! 
        % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
        % of using only a part of unlabeled data searching for query points
        %% PREPARE KERNEL MATRIX
        Klq = Klu;
        Kqq = Kuu;
        Kuq = Kuu;
        K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
        n_q = n_u; 
        %% CONSTANTS
        ONEA = ones(n_l+n_u+n_q,1);
        ONEL = ones(n_l,1);
        ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
        ONEU = ones(n_u,1);
        ONEQ = ones(n_q,1);
        %% PARAMTERS THAT ARE FIXED IN THIS OPTIMIZATION: q,AMatrix
        qc = q; % a copy of Q
        r_q=[ONEL;1-qc;qc];
        p_q=[zeros(n_l+n_u,1);qc];
        
        KA= K.*AMatrix;
        KApinv = pinv(KA);
        
        cvx_begin 
            variable   beta_Var(n_l+n_u+n_q,1)
            variable   theta_Var(n_l+n_u+n_q,1)
            variable   dtVar
            %% DECLARE OPTIMIZATION TO CVX AS A SOCP
            minimize (1/2*lambda*dtVar+beta_Var'*(r_q+p_q)+theta_Var'*p_q)
            subject to
               theta_Var+beta_Var-ONED>=0;
               beta_Var>=0;
               theta_Var'*KApinv*theta_Var<=dtVar;
        cvx_end 
        %% RETURNING RESULTS: beta_var,eta_var
        beta_Varc=beta_Var;
        eta_Varc =theta_Var+beta_Var-ONED;
end
%% This function calculates A the lable matrix using SDP formulation
%  it has two SDP constraint as the original 
function [AMatrix] = ADMM_Label01(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var)
    
    bSize  = 2;
    g = 0; % For now! 
    % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
    % of using only a part of unlabeled data searching for query points
    Klq = Klu;
    Kqq = Kuu;
    Kuq = Kuu;
    n_q = n_u; 
    
    ONEA = ones(n_l+n_u+n_q,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);
    qc = q; % a copy of Q

    cvx_begin sdp
        variable   A_uu(n_u,n_u)
        variable   A_uq(n_u,n_q)
        variable   A_lu(n_l,n_u)
        variable   tVar
        expression K
        expression A
        expression r_q
        expression p_q
        r_q=[ONEL;1-qc;qc];
        p_q=[zeros(n_l+n_u,1);qc];
        K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
        % This is the lable matrix which we want to estimate it's
        % unlabeled part
        A = [Yl*Yl',A_lu,Yl*ONEQ';A_lu',A_uu,A_uq;ONEQ*Yl',A_uq',ONEQ*ONEQ'];

        minimize (tVar)
        subject to 
           A>=0
           diag(A_uu)==ONEU
           eta_Var>=0
           beta_Var>=0 
           [K .* A,ONED-beta_Var+eta_Var;(ONED-beta_Var+eta_Var)',2*(tVar-beta_Var'*r_q-eta_Var'*p_q)/lambda ]>=0;
    cvx_end 

    AMatrix = A; 

end
%% This code computes Lable Matrix regulating it's norm in order to minimize the rank of it
%% function [AMatrix,optval,dualVars] = ADMM_Label03(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var)
%   This function is deleted from this file it is in ActiveCRSemiSVM02.m 
% function [AMatrix,optval] = ADMM_Label031(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var,rho)
%   This function is deleted from this file it is in ActiveCRSemiSVM02.m     
%% Fix beta,eta and Optimize A (Label Matrix)    
function [AMatrix,optval,dualVars] = ADMM_Label032(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var)
    bSize  = 2;
    r = 25;
    g = 0; % For now! 
    % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
    % of using only a part of unlabeled data searching for query points
    %% Prepare KERNEL MATRIX
    Klq = Klu;
    Kqq = Kuu;
    Kuq = Kuu;
    K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
    n_q = n_u; 
    n_luq = n_l+n_u+n_q;
    %% CONSTANTS
    ONEA = ones(n_luq,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);
    e_lq =[ones(n_l,1);zeros(n_u,1);ones(n_q,1)];
    I_u  =zeros(n_luq,n_luq); 
    I_u(n_l+1:n_l+n_u,n_l+1:n_l+n_u)=eye(n_u);
    Y_lq = [Yl;zeros(n_u,1);ONEQ];
    YY_lq = Y_lq*Y_lq';
    ALLA = ONEA*ONEA';
    CM =e_lq*e_lq'+I_u;
    C_lq = YY_lq+I_u;
    ONE_lq = [ONEL;zeros(n_u,1);ONEQ];
    %% FIXED VARIABLES IN THIS OPTIMIZATION: eta,beta,q
    tau = ONED-beta_Var+eta_Var;
    qc = q;
    r_q=[ONEL;1-qc;qc];
    p_q=[zeros(n_l+n_u,1);qc];
    tau = ONED-beta_Var+eta_Var;
    m=2/lambda*(beta_Var'*r_q+eta_Var'*p_q);
    %% CVX
    cvx_precision low
    cvx_begin sdp
        variable   A_uu(n_u,n_u) symmetric 
        variable   A_uq(n_u,n_q) 
        variable   A_lu(n_l,n_u)
        variable   A_lq(n_l,n_q)
        variable   A_qq(n_q,n_q) symmetric
        variable   A_ll(n_l,n_l) symmetric
        variable   tVar
        dual variable drho; 
        dual variable U_uu;
        dual variable Z;
        dual variable T;
        dual variable U;
        expression B
        expression A 
        % This is the lable matrix which we want to estimate it's
        % unlabeled part
%         A = [Yl*Yl',A_lu,Yl*ONEQ';A_lu',A_uu,A_uq;ONEQ*Yl',A_uq',ONEQ*ONEQ'];               
        A = [A_ll,A_lu,A_lq;A_lu',A_uu,A_uq;A_lq',A_uq',A_qq]; % LABEL MATRIX AS A WHOLE
        B = [Yl*Yl',Yl*ONEQ';ONEQ*Yl',ONEQ*ONEQ'];   
        %% DECLARE OPTIMIZATION PROBLEM TO CVX
        minimize (tVar)
        subject to 
           T:A>=0;
           U:[A_ll,A_lq;A_lq',A_qq]==B;
           Z:[K .* A,ONED-beta_Var+eta_Var;(ONED-beta_Var+eta_Var)',2*tVar/lambda-m ]>=0;
           U_uu:diag(A_uu)==ONEU;
           drho: norm(A,'fro')<=r;
    cvx_end 
    %% RETURN RESULTS
    optval = cvx_optval;
    AMatrix = A; 
    lind = 1:n_l;  qind=n_l+1:n_l+n_q;
    U2=[U(lind,lind),zeros(n_l,n_u),U(lind,qind);zeros(n_u,n_l),diag(U_uu),zeros(n_u,n_q);U(qind,lind),zeros(n_u,n_u),U(qind,qind)];
    U2 = -U2;% !!! CVX returns equality constraint parameters negatively
    dualVars.drho=drho;
    dualVars.U_uu = U_uu;
    dualVars.U = U2; % !!! CVX returns equality constraint parameters negatively 
    dualVars.Z=Z;
    dualVars.T=T;
    sn=size(dualVars.Z,1)-1;
    X=dualVars.Z(1:sn,1:sn);
    x=dualVars.Z(1:sn,sn+1);
    
end
%% This function doesnot worked correctly and I don't know why?!!!
function [AMatrix,optval] = ADMM_Label041(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var,dualVars)
    rho = dualVars.rho;
    bSize  = 2;
    r = 25;
    g = 0; % For now! 
    % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
    % of using only a part of unlabeled data searching for query points
    Klq = Klu;
    Kqq = Kuu;
    Kuq = Kuu;
    n_q = n_u; 
    n_luq=n_l+n_u+n_q;
    
    ONEA = ones(n_l+n_u+n_q,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);
    K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
    tau = ONED-beta_Var+eta_Var;
    e_lq =[ones(n_l,1);zeros(n_u,1);ones(n_q,1)];
    I_u  =zeros(n_luq,n_luq); 
    I_u(n_l+1:n_l+n_u,n_l+1:n_l+n_u)=eye(n_u);
    Y_lq = [Yl;zeros(n_u,1);ONEQ];
    YY_lq = Y_lq*Y_lq';
    ALLA = ONEA*ONEA';
    CM =e_lq*e_lq'+I_u;
    C_lqu = YY_lq+I_u;
    normClqu = norm(C_lqu,'fro')

    
    stepsize = 0.1;
    epsgrad = 0.1;
    rhogradz= 1;
    qc = q; % a copy of Q
    r_q=[ONEL;1-qc;qc];
    p_q=[zeros(n_l+n_u,1);qc];

    m=2/lambda*(beta_Var'*r_q+eta_Var'*p_q);

    cvx_begin sdp
        variable   U(n_luq,n_luq) symmetric
        variable   x(n_luq,1)
        variable   tb
        X = semidefinite(n_luq);
        T = semidefinite(n_luq);

        minimize (1/(4*rho)*square(tb)+trace(U'*C_lqu)+2*x'*tau+rho*r-1/2*lambda*m)
        subject to 
           [X,x;x',lambda/2]>=0;
           norm(T+X.*K-U.*CM,'fro')<= tb;
           T+X.*K-U.*CM >=0;
    cvx_end 
    optval = cvx_optval;
    AMatrix =1/(2*rho)*(T+X.*K-U.*CM); 

    rhogradz = -norm(T+X.*K-U.*CM,'fro')^2/(4*rho^2)+r;
    rhonext = 1/2*norm(T+X.*K-U.*CM,'fro')*sqrt(1/r);
    %rho  = rhonext;
        
%         if rhogradz < 0
%             rho = rho+stepsize;
%         else
%             rho = rho-stepsize;
%         end
        
   
end
%% Optimization based on converting two sdp constraints to one. The problem is I don't know how to find the primal variables
function [AMatrix,optval] = ADMM_LabelSDPConstrReduction(Kll,Klu,Kuu,Yl,n_l,n_u,lambdal,q,beta_Var,eta_Var,AMatrix)
%% Optimize :max_(?,X)f(X,?)= m+(2?)^(1/2) (<X,D_?k>-?)^(1/2)-<X,D_lq>-<X,D_u> 
% This objective function is the dual of the original ADMM_Label Objective
% by removing one sdp constraint of the original problem. 
% This method is converts two sdp constriants in original problem to one problem and the uses a method
% based on the paper "A Hybrid algorithm for convex semidefinite Optimiztation". 
%% DEFINE CONSTANTS
[bSize,r,g,K,tau,set_Dl,set_Du,set_Dq]=defineConsts(Kll,Klu,Kuu,Yl,n_l,n_u,lambdal,q,beta_Var,eta_Var);
% ??^T?K=  U D_{?k}^(+-) U^T
% D_?k=max {0,D_?k^(+,-)}
% D_lq :D_lq=2?_(i,j?D_l?D_Q,i<j)E_ij/(U^T E_ij U) y_i* y_j % 
% Is this coefficient(2) is correct?
% D_u  :D_u=2?_(i?D_u)E_ii/(U^T E_ii U)  
% D_u  is equal to I_u;
%% DEFINE GLOBAL VARIABLES
global lambda
global C_D
global D_luq
global D_u
global theta
global V_i 
global v_i
%% 
lambda=lambdal;
n_q = size(set_Dq,2);
TA=tau*tau';
% TAK=TA./K;
% [U,D_TKMN]=eig(TAK);
% C_D2=U*D_TKMN*U';
% norm(TAK-C_D2,'fro')
% Zd=zeros(size(K,1));
% D_TK = bsxfun(@max,D_TKMN,Zd); %  D_TK=max(D_TKMN,0) 
% C_D=U*D_TK*U';
% norm(C_D-C_D','fro')
% C_D=(C_D+C_D')/2;
% p=minsgreater(AMatrix,C_D);
% it's better to make near zero elements of U equal to zero
%% Constructing Constant Matrix D_lq
C_D=TA;
Yset=[Yl;zeros(n_u,1);ones(n_q,1)];
E_ij=zeros(size(K,1));
D_luq=Yset*Yset';
D_luq(n_l+1:n_l+n_u,n_l+1:n_l+n_u)=eye(n_u); % make diagonal elements of u equal to 1 
D_luq=D_luq .* K;
c=sqrt(2*lambda);
%% OPTIMIZE THIS SIMPLER PROBLEM USING CVX IN ORDER TO COMPARE RESULTS
n = size(D_luq,1);
cvx_begin
    variable X(n,n) semidefinite
    minimize( trace(X'*D_luq)-c*sqrt(trace(X'*C_D)) ) 
cvx_end
%% OPTIMIZE THE SAME PROBLEM USING METHOD OF PAPER "A Hybrid ....."
theta  =0 ; % we will prove that optimal theta is 0
%% These parameters are for minFunc
maxFunEvals = 25;
options = [];
options.display = 'none';
options.maxFunEvals = maxFunEvals;
options.Method = 'lbfgs';
% now we must find a vector v_0 for initialization such that v_0=argmin<v_0*v_0',D_lq+D_u>
% this is equal to eigenvector corresponding to min eigen value. 
%% Initial point using smallest eigenvector 
[v_0,d_0]=eigs(D_luq,1); % the smalleset eigenvector
f_i= 10000000;% Inf 
V_i=v_0;
theta = 0 ;
%% Starting optimization loop
converged  = false;
while ~converged
    %% compute v_i= approxEV(-grad(f(V_i*V_i',eps))
    % I must write function grad_f. 
    [v_i,e_v] = eigs(grad_f(V_i,eps,lambda,C_D,D_luq,theta),1);
    %% solve min_c1,c2   f(c1*V_i*V_i'+c2*v_i*v_i') s.t.
        % c1,c2 >=0
    X_i=V_i*V_i';
    X_1=v_i*v_i';
    c1= trace(X_i*D_luq');
    c2=trace(X_1*D_luq');
    c3=trace(X_i*C_D');
    c4=trace(X_1*C_D');
    g_luq=[c1,c2]';
    g_CD=[c3,c4]';
    c=minObjective(lambda,g_luq,g_CD);
    %[c,f_ip,exitflag,str]=minFunc(@f_minComb2d,[0.1 0.1]',options);
    %[c,f_ip,exitflag,str]=minFunc(@f_combDirection,[0.1 0.1]',options);
    %% UPDATE V_ip
    V_ip = [sqrt(c(1))*V_i,sqrt(c(2))*v_i];
    %set V_(i+1)=[sqrt(\alpha).V_i,sqrt(\beta).v_i]
    
    % run nonlinear update,improve V_(i+1) by finding a local minimum of
        % f(V*V') wrt V starting with V_(i+1)
    %    V_(i+1) = minFunc()
    if abs(f_ip-f_i)<eps  , converged  = true; end
    f_i = f_ip;
    V_i = V_ip;
% until approximate guarantee has been reached
end
X = V_i*V_i';
%% HOW TO COMPUTE PRIMAL VARIABLES?...

end
function [c]=minObjective(lambda,gu,gc)
gamma=min(gu(1)/gc(1),gu(2)/gc(2));
cst=sqrt(2*lambda);
t=cst/(2*gamma);
end
function    [ObVal,df,ddf]=f_minComb2d(d)
    global V_i
    global v_i
    global C_D
    global D_luq
    %global theta
    global lambda

    X_i=V_i*V_i';
    X_1=v_i*v_i';
    c1= trace(X_i*D_luq');
    c2=trace(X_1*D_luq');
    c3=trace(X_i*C_D');
    c4=trace(X_1*C_D');
    g_luq=[c1,c2]';
    g_CD=[c3,c4]';

    ObVal = g_luq'*d-sqrt(2*lambda)*sqrt(g_CD'*d);
    df = g_luq-0.5*sqrt(2*lambda)*g_CD/sqrt(g_CD'*d);
    ddf =0.25*sqrt( 2*lambda)*g_CD*g_CD'/(sqrt(g_CD'*d)*(g_CD'*d));
end
%% This function computes objective function, also gradient and hessian of it
function    [ObVal,df,ddf]=f_combDirection(d)
    global V_i
    global v_i
    global C_D
    global D_luq
    global theta
    global lambda

    X=d(1)*V_i*V_i'+d(2)*v_i*v_i';
    ObVal = sqrt(2*lambda)*sqrt(trace(X'*C_D)-theta)-trace(X'*D_luq);
    df = 1/2 *sqrt( 2*lambda)* C_D/sqrt(trace(X'*C_D)-theta)-D_luq;
    ddf =-0.25*sqrt( 2*lambda)*C_D*C_D/(sqrt(trace(X'*C_D)-theta)*(trace(X'*C_D)-theta));
    % Make it negative because we want to maximize the objective function 
    ObVal = -ObVal;
    df = -df;
    ddf = -ddf;
end
%% gradient of Objective function is ?_X f(X,?)=  1/2 (2?)^(1/2)   D_?k/(<X,D_?k>-?)^(1/(2 ))-D_lq-D_u      
function [G]=grad_f(V_i,eps,lambda,C_D,D_luq,theta)
    X=V_i*V_i';
    G = D_luq-1/2 *sqrt( 2*lambda)* C_D/sqrt(trace(X'*C_D)-theta);
end
%% DEFINING CONSTANTS
function [bSize,r,g,K,tau,set_Dl,set_Du,set_Dq]=defineConsts(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var)
    % this function stores and computes the constants within the calling
    % function. 
    
    bSize  = 2;
    r = 25;
    g = 0; % For now! 
    % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
    % of using only a part of unlabeled data searching for query points
    Klq = Klu;
    Kqq = Kuu;
    Kuq = Kuu;
    n_q = n_u; 
    n_luq=n_l+n_u+n_q;
    
    ONEA = ones(n_l+n_u+n_q,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);
    K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
    tau = ONED-beta_Var+eta_Var;
    set_Dl = 1:n_l;
    set_Dq = n_l+n_u+1:n_luq;
    set_Du = n_l+1:n_u;
end
        
%% THIS FUNCTION DOES NOT WORKED MAY BE THE DUAL PROBLEM DERIVED INCORRECTLY
function [AMatrix] = ADMM_Label07(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,q,beta_Var,eta_Var)
    
    bSize  = 2;
    r = 25;
    g = 0; % For now! 
    % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
    % of using only a part of unlabeled data searching for query points
    Klq = Klu;
    Kqq = Kuu;
    Kuq = Kuu;
    n_q = n_u; 
    n_luq=n_l+n_u+n_q;
    ONEA = ones(n_l+n_u+n_q,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);
    K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
    tau = ONED-beta_Var+eta_Var;
    TTAu=tau*tau';
    e_lq =[ones(n_l,1);zeros(n_u,1);ones(n_q,1)];
    I_u  =zeros(n_luq,n_luq); 
    I_u(n_l+1:n_l+n_u,n_l+1:n_l+n_u)=eye(n_u);
    Y_lq = [Yl;zeros(n_u,1);ONEQ];
    YY_lq = Y_lq*Y_lq';
    ALLA = ONEA*ONEA';
    CM =e_lq*e_lq'+I_u;
    C_lq = YY_lq+I_u;
    %normYYI = norm(C_lq,'fro')

    rho =0.0028;
    h   =42.0;

    qc = q; % a copy of Q
    r_q=[ONEL;1-qc;qc];
    p_q=[zeros(n_l+n_u,1);qc];
    epstol = 0.000001;
    m=2/lambda*(beta_Var'*r_q+eta_Var'*p_q);
    %this piece of code implements the alternating direction of multipliers for SDP problem PQ6 page 74 of my notes
    X=rand(n_luq,n_luq);
    U=rand(n_luq,n_luq);
    X=(X+X')/2;
    err = epstol +1000;
    while err > epstol 
        XP=Prox_Snp(X);
        norm(X-XP,'fro')
        U_CM=2*XP.*K.*CM-2*rho*C_lq;
        norm(U_CM-U_CM','fro')
        XPre=X;
        X=0.5*U_CM./K+rho/(2*h)*TTAu./K./K;
        rhogradz = -norm(2*X.*K-U_CM,'fro')^2/(4*rho^2)+r;
        rho = rhogradz;
        norm(X-X','fro')
        err = norm(X-XPre,'fro')
    end
        
    AMatrix =1/(2*rho)*(2*X.*K-U.*CM); 
end
%% Projection of a matrix on Semidefinite Cone
function XP=Prox_Snp(X)
[V,D]=eig(X);
D_p = D;
for i=1:size(X,1)
    if D_p(i,i)<0
        D_p(i,i)=0;
    end
end
XP = V*D_p*V';
end
%% Global search Only for test
% finds the global optimial to see which instance is the best query.
% it's a good idea to compare classification accuracy when this instance
% queried instead of the algorithm query. 
function [qi,OPTMAT,ZA]=findOptimalWithSearch(Kll,Klu,Kuu,Yl,n_l,n_u,lambda)
    
    bSize  = 1;
    g = 0; % For now! 
    % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
    % of using only a part of unlabeled data searching for query points
    Klq = Klu;
    Kqq = Kuu;
    Kuq = Kuu;
    n_q = n_u; 
    q   = bSize*ones(n_u,1)/n_u;
    ONEA = ones(n_l+n_u+n_q,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);

    OptVal = 100000000000000;
    qi  =0 ;
    for i=1:n_q
        qc = zeros(n_q,1);qc(i)=1;
        cvx_begin
            variable   A_uu(n_u,n_u)
            variable   A_uq(n_u,n_q)
            variable   A_lu(n_l,n_u)
            variable   beta_Var(n_l+n_u+n_q,1)
            variable   eta_Var(n_l+n_u+n_q,1)
            variable   tVar
            expression K
            expression A
            expression r_q
            expression p_q
            r_q=[ONEL;1-qc;qc];
            p_q=[zeros(n_l+n_u,1);qc];
            K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
            % This is the lable matrix which we want to estimate it's
            % unlabeled part
            A = [Yl*Yl',A_lu,Yl*ONEQ';A_lu',A_uu,A_uq;ONEQ*Yl',A_uq',ONEQ*ONEQ'];

            minimize (tVar)
            subject to 
               A>=0;
               diag(A_uu)==ONEU;
               eta_Var>=0;
               beta_Var>=0; 
               [K .* A,ONED-beta_Var+eta_Var;(ONED-beta_Var+eta_Var)',2*(tVar-beta_Var'*r_q-eta_Var'*p_q)/lambda ]     >=0;
        cvx_end
        OPTMAT (i) = cvx_optval;
        ZA = A;
        if cvx_optval < OptVal
            OptVal = cvx_optval;
            qi = i;
        end
    end
end
function [p]=minsgreater(AMatrix,C_D)
p=100;
plow=0;
z=zeros(size(C_D,1),1);
phigh=100;
while plow<phigh
    pMid=(plow+phigh)/2;
    CompMatrix=AMatrix-p*C_D;
    e=eig(CompMatrix);
    c=bsxfun(@lt,e,z);
    if c
        plow = pMid;
    else
        phigh=pMid;
    end
end

end
%% Select subTasks data for Diverse Subtask AL
% This function splits data to T subsets. It splits labeled data
% redundantly,but divides unlabeled data to T disjoint sets. 
function subTaskdata = selectsubtaskData(Kll,Klu,Kuu,Yl,n_l,n_u,lambda,T)
    %% select Labeled; A labeled may be selected more than once is redundant data in tasks 
    % number of  Tasks:T
    muPar = -1; % set to <0 to auto determine the amount of diversity
    sample_pLabeled = 0.8; % random sample percentage (value 1 was used in the experiments for the paper)
    % prepare index of training instances for each SVM 
    % such that every set has atleast one instance with any label
    index_set_Labeled = cell(T,1);
    for t = 1:T
        rp = randperm(n_l);
        rpp = find(Yl==1);
        rpn = find(Yl==-1);
        slp=round(sample_pLabeled*n_l/2);
        sln=floor(sample_pLabeled*n_l)-slp;
        if slp < 1 ,slp=1;end
        if sln < 1 ,sln=1;end
        idxlablepositive= rpp(1:slp);
        idxlablenegative=rpn(1:sln);
        index_set_Labeled{t,1}=[idxlablepositive,idxlablenegative];
        taskn_l(t)=size(index_set_Labeled{t,1},2);
    end
    disp (['train .... with T=', num2str(T), ' mu=', num2str(muPar), ' p=', num2str(sample_pLabeled)])
    %% Select Unlabeled data; each unlabeled data is selected only once
    sample_pUnLabeled = 1/T; % random sample percentage (value 1 was used in the experiments for the paper)
    tUnlbsize=floor(n_u/T);

    index_set_UnLabeled = cell(T,1);
    rp = randperm(n_u);
    st=1;
    for t = 1:T
        en=st+tUnlbsize-1;
        if en > n_u, en=n_u; end
        if t==T, en=n_u;end % we must select all unlabeled data 
        index_set_UnLabeled{t,1} = rp(st:en);
        taskn_u(t)=size(index_set_UnLabeled{t,1},2);
        st=en+1;
    end
    %% prepare Kernels for each tasks
    task_Kll = cell(T,1);
    task_Klu = cell(T,1);
    task_Kuu = cell(T,1);
    for t=1:T
        lbindex=index_set_Labeled{t,1};
        ulbindex=index_set_UnLabeled{t,1}
        task_Yl{t,1}=Yl(lbindex);
        task_Kll{t,1}=Kll(lbindex,lbindex);
        task_Klu{t,1}=Klu(lbindex,ulbindex);
        task_Kuu{t,1}=Kuu(ulbindex,ulbindex);
    end
    subTaskdata.index_set_UnLabeled=index_set_UnLabeled;
    subTaskdata.index_set_Labeled=index_set_Labeled;
    subTaskdata.task_Kll=task_Kll;
    subTaskdata.task_Klu=task_Klu;
    subTaskdata.task_Kuu=task_Kuu;
    subTaskdata.T  = T;
    subTaskdata.Kll=Kll;
    subTaskdata.Klu=Klu;
    subTaskdata.Kuu=Kuu;
    subTaskdata.n_l=n_l;
    subTaskdata.n_u=n_u;
    subTaskdata.Yl=Yl;
    subTaskdata.taskn_l=taskn_l;
    subTaskdata.taskn_u=taskn_u;
    subTaskdata.task_Yl=task_Yl;
end
%% learning and active learning for all subTasks
function [q,model_set,Y_set]=diverseSubTaskAL(subTaskdata,trainOptions)
%% Initializing Q: the subset of unlabeled data to query from 
    n_u = subTaskdata.n_u; % consider all of unlabeled data for selecting query
    n_q=n_u;
    n_l=subTaskdata.n_l;
    Yl=subTaskdata.Yl;
    bSize  = 1;
    g = 0; % For now! 
%% Initialize some constants    
    ONEQ = ones(n_q,1);
%% Initalize starting point of (query,Lables)for Alternating Minimization      
    q   = bSize*ones(n_u,1)/n_u;
    % initialize label for unlabled data randomly
    % use it for making label matrix A 
    Yu =sign(-1+2*rand(n_u,1));
    Ya = [Yl;Yu;ONEQ];
    AMatrix = Ya*Ya';
%% Starting Alternating minimization Loop for query and (Learning,Lables)    
    for i=1:4 
%% Fix q and Optimize Semi-supervised SVM(Learning,Lables)
        qc = q;
        [model_set,Y_set]= trainDiverseSubTask(subTaskdata,trainOptions,qc,AMatrix);
%% Fix (Learning,Lables) and Optimizing to find q: query point(s)
        % the following function will do the same work
        [q]=OptQuerySubTask(subTaskdata,trainOptions,model_set,Y_set,q,bSize);
    end
end
%% train Diverse subTask Semi-supervised Learning
function [model_set,Y_set]= trainDiverseSubTask(subTaskdata,trainOptions,qc,AMatrix)
T=subTaskdata.T;
n_l=subTaskdata.n_l;
n_u=subTaskdata.n_u;
n_q=n_u;
n=n_l+n_u+n_q;

%% initialize 
model_set=cell(T,1);
indep=true;
linearterm=0;w0_norm=0;
linearterm = zeros(n,T);
alpha_set = zeros(n,T);
Y_set  =zeros(n,T);
Kll=subTaskdata.Kll;Klu=subTaskdata.Klu;Kuu=subTaskdata.Kuu;Klq=Klu;Kuq=Kuu;Kqq=Kuu;
K=[Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
%% train each task independently
lambda_t=1;
muParInd=0;
for t=1:T
    [model_set{t},~]=train_divtask(t,subTaskdata,Y_set(:,t),trainOptions,qc,indep,model_set,linearterm,w0_norm);
    lambda_t = 1;
    ind      = indexSet(t,subTaskdata);
    alpha_set(ind,t) = model_set{t}.alpha_t;
    ind_l    = indexSet(t,subTaskdata,1);
    Y_set(ind,t)  = estimateY(t,subTaskdata,model_set{t});
    % set estimate y for labeled data equal to given Yl
    Yl = subTaskdata.task_Yl{t,1};
    Y_set(ind_l,t) = Yl;
end
lambda_t=1;
%% train all tasks simultaneously
muPar=trainOptions.muPar;
converged=false;
iter=1;
indep=false;
while ~converged && iter < 20  
    for t=1:T
        ind=indexSet(t,subTaskdata);
        % computing linearterm 
        [linearterm]=complinearterm(t,alpha_set,Y_set,subTaskdata,model_set);
        [model_set{t},diff(t)]=train_divtask(t,subTaskdata,Y_set(ind,t),trainOptions,qc,indep,model_set{t},linearterm,muPar);
        alpha_set(ind,t) = model_set{t}.alpha_t;
    end
    sdiff(iter)=sum(diff);
    if sdiff(iter) < 0.01
        converged = true;
    end
    iter = iter + 1;
end
%% send information to above em optimization.
% model_set 
% Y_set

end
%% train a single subtask either independently or simulataneously
function [model_set,diff]=train_divtask(t,subTaskdata,Ypre,trainOptions,qc,indep,model_set,linearterm,muPar)

rhoPar=1;
%% extract subtask data:n_l,n_u,Kll,Klu,Kuu,Yl
n_l=subTaskdata.taskn_l(t);
n_u=subTaskdata.taskn_u(t);
n_q=n_u;
n=n_l+n_u+n_q;
Kll=subTaskdata.task_Kll{t,1};
Klu=subTaskdata.task_Klu{t,1};
Kuu=subTaskdata.task_Kuu{t,1};
Yl =subTaskdata.task_Yl{t,1};
q_t=qc(subTaskdata.index_set_UnLabeled{t,1});
Klq=Klu;Kuq=Kuu;Kqq=Kuu;
K_tt=[Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
T=subTaskdata.T;
tTask=t;
%% is it independently training or simultaneously?
if indep
    %% directly optimize with respect to R_t,beta_t,eta_t
    % qc ?
    muPar=0;
    lambda_t=1;
    d_t=0%lambda_t;
    flag=false;
    iter=1;
    c_t=1/(lambda_t);
    [R_t,beta_t,eta_t,alpha_t]=OptimizeR_beta_eta(n_l,n_u,q_t,K_tt,Yl,c_t,d_t,false);
    model_set.eta_t=eta_t;
    model_set.beta_t=beta_t;
    model_set.R_t=R_t;
    model_set.lambda_t=lambda_t;
    model_set.alpha_t = alpha_t;
%     [alpha_t]=compalpha(t,subTaskdata,model_set,muPar,lambda_t,qc);   
    %% assume that lambda_t is given. In fact the result is converges towards zero otherwise
%     while ~flag && iter < 2
%         alpha_pre=alpha_t;
%         lambda_t = 1;%max(0,sqrt(alpha_t'*(K_tt.*R_t)*alpha_t)/2-(T-1)*muPar);
%         c_t=1/((subTaskdata.T-1)*muPar+lambda_t);
%         [R_t,beta_t,eta_t]=OptimizeR_beta_eta(n_l,n_u,q_t,K_tt,Yl,c_t,d_t,true,R_t);
%         model_set.eta_t=eta_t;
%         model_set.beta_t=beta_t;
%         model_set.R_t=R_t;
%         [alpha_t]=compalpha(t,subTaskdata,model_set,muPar,lambda_t,qc);
%         %%%%%%%%%%%%%%%%
%         update the lambda, what is alpha? 
%         lambda = max(0,sqrt(alpha_t'*(K_tt.*R_t)*alpha_t-4*muPar*alpha'*linearterm+4*muPar^2*w0_norm)/2-(T-1)*muPar);
%         
%         iter=iter+1;
%         if norm(alpha_t-alpha_pre)<10^(-5)
%             flag=true;
%         end
%    end
else
    lambda_t=model_set.lambda_t;
    c_t=1/((T-1)*trainOptions.muPar+lambda_t);
    d_t=0;% assume it is zero, i.e. we don't want to learn lambda_t
    %% initialize Yprime equal to Y_t from independent training of subtask
    Yprime = Ypre;% 
    R_t    = model_set.R_t;
    beta_t = model_set.beta_t;
    eta_t  = model_set.eta_t;
    n_t=size(Yprime,1);
    %% Start ADMM process
    z_Y = -1+rand(n_t,1)*2; % generate random numbers beween -1 to 1
    Z_R = z_Y*z_Y';
    z_beta=zeros(n,1);
    z_eta =zeros(n,1);
    flag=false;
    iter_k=1;
    beta_tpre=model_set.beta_t;
    eta_tpre=model_set.eta_t;
    R_tpre=R_t;
    % in the following ADMM,first I set the rhopar to 0.1, the algorithm
    % does not converged. But after, I set the rhopar to 1, the algorithm
    % converged very rapidly!( although it tested when mupar was 0).
    while iter_k < 20 && ~flag 
        %% Optimize with respect to R_t
        [model_set.R_t,model_set.beta_t,model_set.eta_t,model_set.alpha_t] =...
            OptOnR_tbetaeta(n_l,n_u,Yl,Yprime,R_tpre,beta_tpre,eta_tpre,K_tt,linearterm,Z_R,z_beta,z_eta,q_t,d_t,c_t,rhoPar,muPar);
%         %% Optimize with respect to beta,eta
%         [beta_t,eta_t]=OptOnbetaeta(n_l,n_u,Yl,Yprime,beta_tpre,eta_tpre,R_t,K_tt,linearterm,Z_R,z_beta,z_eta,q_t,d_t,c_t,rhoPar,muPar);
        
        %% Optimize with respect to Y_t
        [Y_t] = estimateY(t,subTaskdata,model_set); 
        % OptOnY_t(n_l,n_u,R_t,Yprime,beta_t,eta_t,K_tt,linearterm,z_Y,q_t,d_t,c_t,rhoPar,muPar);
        
        %% Update Lagrange multipliers z,Z
        z_Y = z_Y+rhoPar*(Yprime-Y_t);
        Z_R = Z_R+rhoPar*(model_set.R_t-R_tpre);
        z_beta = z_beta+rhoPar*(beta_t-beta_tpre);
        z_eta  = z_eta +rhoPar*(eta_t-eta_tpre); 
        rn(1,iter_k)= norm(model_set.R_t-R_tpre,'fro')/norm(model_set.R_t,'fro');
        rn(2,iter_k)= (norm(model_set.beta_t-beta_tpre)+norm(model_set.eta_t-eta_tpre))/(norm(beta_tpre)+norm(eta_tpre));
        rn(3,iter_k)= norm(Y_t-Yprime);
        %% Update Yprime equal to largest eigen vector of R_t
        Yprime = Y_t;%estimateY(t,subTaskdata,model_set);
        rhoPar = 1.2*rhoPar;
%         if norm(model_set.R_t-R_tpre,'fro')/norm(model_set.R_t,'fro')<0.05 
%             flag = true;
%         end
        if rn(2,iter_k) < 0.005
            flag = true;
        end
        beta_tpre= model_set.beta_t;
        eta_tpre = model_set.eta_t;
        R_tpre   = model_set.R_t;
        iter_k   = iter_k +1;
    end
end
if ~indep
    diff=rn(2,iter_k-1);
else
    diff=0;
end
lambda_t=max(0,1);% this is incorrect, what must be it's value?

end
%% Optimize Query 
function [q]=OptQuerySubTask(subTaskdata,trainOptions,model_set,Y_set,q,bSize)
    T      = subTaskdata.T;
    n_u    = subTaskdata.n_u;
    coefmat= zeros(n_u,T);
    upperbound= ones(n_u,T);
    lowerbound = zeros(n_u,T);
    for t=1:T
        n_lt = subTaskdata.taskn_l(t);
        n_ut = subTaskdata.taskn_u(t);
        n_qt = n_ut;
        n_t  = n_lt+n_ut+n_qt;
        eta_qt = model_set{t}.eta_t(n_lt+n_ut+1:n_t);
        beta_qt= model_set{t}.beta_t(n_lt+n_ut+1:n_t);
        beta_ut= model_set{t}.beta_t(n_lt+1:n_lt+n_ut);
        ind    = subTaskdata.index_set_UnLabeled{t};
        upperbound(ind,t) = ones(n_ut,1)-model_set{t}.alpha_t(n_lt+1:n_lt+n_ut);
        lowerbound(ind,t) = abs(model_set{t}.alpha_t(n_lt+n_ut+1:n_t));
        
        coefmat(ind,t)= eta_qt+beta_qt-beta_ut; 
    end
    
    cvx_begin
        variable q(n_u)
        minimize sum(coefmat'*q)
           subject to 
                 0<=q
                 sum(q)==bSize
                 for t=1:T
                     q>= lowerbound(:,t);
                     q<= upperbound(:,t);
                 end
    cvx_end 
end
%% Optimize ADMM objective function,fixed R_t,with respect to Y_t,beta_t,eta_t
function [Y_tr]= OptOnY_t(n_l,n_u,R_t,Yprime,beta_t,eta_t,K_tt,linearterm,z_Y,q_t,d_t,c_t,rhoPar,muPar)
    %% Initializing Q: the subset of unlabeled data to query from 
        lambda=2; %For now but it must be specified by the caller function 
        n_q = n_u;
    %% Initialize some constants    
        ONED = [ones(n_l+n_u,1);zeros(n_q,1)]; 
        ONEL = ones(n_l,1);
        pq_t=[zeros(n_l+n_u,1);q_t];
        rq_t=[ONEL;1-q_t;q_t];
   
        cvx_begin sdp
            variable   Y_t(n_l+n_u+n_q,1) %
            variable   tVar
                       
            minimize (tVar+z_Y'*(Yprime-Y_t)+1/2*rhoPar*square_pos(norm(Yprime-Y_t)))
            subject to 
               [K_tt .* R_t, ONED+muPar*c_t*linearterm*Y_t-beta_t+eta_t;...
                (ONED+muPar*c_t*linearterm*Y_t+-beta_t+eta_t)',...
                (c_t*tVar-beta_t'*rq_t-eta_t'*pq_t-d_t)/lambda ]>=0
               Y_t<=1
               Y_t>=-1
        cvx_end     

        Y_tr = Y_t;
end
%% Optimize ADMM objective function with respect to R_t
function [R_t,beta_t,eta_t]= OptOnR_t(n_l,n_u,Yl,Y_t,R_tpre,beta_tpre,eta_tpre,K_tt,linearterm,Z_R,z_beta,z_eta,q_t,d_t,c_t,rhoPar,muPar)
   %% Initializing Q: the subset of unlabeled data to query from 
        lambda=2; %For now but it must be specified by the caller function 
        n_q = n_u;
        n=n_l+n_u+n_q;
   %% Initialize some constants    
        ONED = [ones(n_l+n_u,1);zeros(n_q,1)]; 
        ONEL = ones(n_l,1);
        ONEQ = ones(n_q,1);
        ONEQMATRIX = ONEQ*ONEQ';
        ONEU = ones(n_u,1);
        pq_t=[zeros(n_l+n_u,1);q_t];
        rq_t=[ONEL;1-q_t;q_t];
   %% minimize with respect to R_t
        cvx_begin sdp
            variable   tVar
            variable   Rt_uu(n_u,n_u) symmetric
            variable   Rt_uq(n_u,n_q)
            variable   Rt_lu(n_l,n_u)
            expression R_t
            % This is the lable matrix which we want to estimate it's
            % unlabeled part
            R_t = [Yl*Yl',Rt_lu,Yl*ONEQ';Rt_lu',Rt_uu,Rt_uq;ONEQ*Yl',Rt_uq',ONEQMATRIX];           
            
            minimize (tVar+trace(Z_R'*(R_t-R_tpre))+1/2*rhoPar*square_pos(norm(R_t-R_tpre)))
            subject to 
               [K_tt .* R_t, ONED+muPar*c_t*linearterm*Y_t-beta_tpre+eta_tpre;...
                (ONED+muPar*c_t*linearterm*Y_t-beta_tpre+eta_tpre)',...
                (c_t*tVar-beta_tpre'*rq_t-eta_tpre'*pq_t-d_t)/lambda ]>=0
               R_t==semidefinite(n);
               diag(Rt_uu)==ONEU
        cvx_end    
end
function [beta_t,eta_t]=OptOnbetaeta(n_l,n_u,Yl,Y_t,beta_tpre,eta_tpre,R_t,K_tt,linearterm,Z_R,z_beta,z_eta,q_t,d_t,c_t,rhoPar,muPar)
    %% Initializing Q: the subset of unlabeled data to query from 
        lambda=2; %For now but it must be specified by the caller function 
        n_q = n_u;
        n=n_l+n_u+n_q;
    %% Initialize some constants    
        ONED = [ones(n_l+n_u,1);zeros(n_q,1)]; 
        ONEL = ones(n_l,1);
        ONEQ = ones(n_q,1);
        ONEQMATRIX = ONEQ*ONEQ';
        ONEU = ones(n_u,1);
        pq_t=[zeros(n_l+n_u,1);q_t];
        rq_t=[ONEL;1-q_t;q_t];
    %% minimize with respect to beta, eta
        cvx_begin sdp
            variable   beta_t(n_l+n_u+n_q,1)
            variable   eta_t(n_l+n_u+n_q,1)
            variable   tVar
            % This is the lable matrix which we want to estimate it's
            % unlabeled part
                       
            minimize (tVar+z_beta'*(beta_t-beta_tpre)+z_eta'*(eta_t-eta_tpre)+1/2*rhoPar*square_pos(norm(beta_t-beta_tpre))+1/2*rhoPar*square_pos(norm(eta_t-eta_tpre)))
            subject to 
               [K_tt .* R_t, ONED+muPar*c_t*linearterm*Y_t-beta_t+eta_t;...
                (ONED+muPar*c_t*linearterm*Y_t-beta_t+eta_t)',...
                (c_t*tVar-beta_t'*rq_t-eta_t'*pq_t-d_t)/lambda ]>=0
               eta_t>=0
               beta_t>=0 
        cvx_end     
end
function [R_t,beta_t,eta_t,alpha_t]=OptOnR_tbetaeta(n_l,n_u,Yl,Y_t,R_tpre,beta_tpre,eta_tpre,K_tt,linearterm,Z_R,z_beta,z_eta,q_t,d_t,c_t,rhoPar,muPar)
    n_q = n_u;
%% Initialize some constants  
    lambda=1;
    n    = n_l+n_u+n_q;
    
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    r_q=[ONEL;1-q_t;q_t];
    p_q=[zeros(n_l+n_u,1);q_t];
    
    M=zeros(n+1,n+1);
    M(n+1,n+1)=lambda/2;
    M(1:n,n+1)=-r_q;
    M         =(M+M')/2;
    ind=[1:n_l,n_l+n_u+1:n];
    ym=zeros(n,1);
    ym(1:n_l)=Yl;
    ym(n_l+n_u+1:n)=ones(n_q,1);
    d_Y = ONED+linearterm*Y_t;
%% Call mosekOptRbetaeta
    [optobjval,R_t,beta_t,eta_t,alpha_t]=mosekOptRbetaeta(n_l,n_u,n_q,p_q,r_q,M,ym,K_tt,d_Y,c_t);

end
%% Compute alpha based on eta , beta , R_t
function [alpha_t]=compalpha(beta_t,nu_t,eta_t,mu_t,r_q,p_q,alpha_t1)
    myeps=0.000005;
    n=size(beta_t,1);
    alpha_t=zeros(n,1);
    Valrv = mu_t-p_q;
    Valpv = r_q-nu_t;
    norm(Valrv-Valpv)
    diff=Valrv-Valpv;
    if abs(diff) < myeps*ones(n,1)
        alpha_t = Valrv;
    else
        for i=1:n
            valr = mu_t(i)-p_q(i);
            valp = r_q(i)-nu_t(i);
            %abs(valr-valp)
            if abs(valr-valp)<myeps  % if this value is correct,
                alpha_t(i)= valr;
            else
               mbe= max(eta_t(i),beta_t(i)); 
               if beta_t(i)>myeps && eta_t(i)>myeps % beta_t(i)==0 and eta_t(i)==0
                        display('error: beta and eta are both greater than eps(zero)');
                        %alpha_t(i)=-10;% some very wrong value
               end
               if eta_t(i)==mbe       % beta_t(i)==0 and eta_t(i)~=0
                 alpha_t(i)=-p_q(i);
               end
               if beta_t(i)==mbe      % beta_t(i)~=0 and eta_t(i)==0
                    alpha_t(i)=r_q(i);
               end
            end
            abs(alpha_t(i)-alpha_t1(i))
        end
    end
end
%% Compute Y_t based on the R_t,(Or R_t,eta_t,beta_t)
function [Y]=estimateY(t,subTaskdata,model_set,sgn)
    if nargin==3, sgn=false; end;
    [V,D]=eigs(model_set.R_t,[],1);
    Y=sqrt(D)*V;
    if (sign(Y(end))~=1),Y=-Y;end% if query points label is negative it means that eigs computed the negative of the labels
    if sgn
        Y_r=sign(Y_r);% test with and without this term 
    end
end
%% Compute Linear Term
function [S]=complinearterm(t,alpha_set,Y_set,subTaskdata,model_set)
s=0;
Kll=subTaskdata.Kll;
Klu=subTaskdata.Klu;
Kuu=subTaskdata.Kuu;
Klq=Klu;Kuq=Kuu;Kqq=Kuu;
K=[Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
ind_t=indexSet(t,subTaskdata);
for r=1:subTaskdata.T
        if r==t ,continue,end;
        ind_r   = indexSet(r,subTaskdata);
        Y_r=Y_set(ind_r,r);
        diag_Yr=diag(Y_r);% this is incorrect. we must use Y_r for all data not just unlabeled data
        K_rt=K(ind_r,ind_t);
        s_r=alpha_set(ind_r,r)'*diag_Yr* K_rt;
        if  abs(max(s_r))> 1
            disp(['error elements of this array must not be larger than 1:',s_r]);
        end
        s=s+s_r;
end
S=diag(s);
end
%% index set of a task
function [ind]=indexSet(t,subTaskdata,part)
    if nargin== 2, part=3,end;
    switch part
        case 1 
            indlb = subTaskdata.index_set_Labeled{t,1};
            ind = indlb;
        case 2 
            indUn = subTaskdata.index_set_UnLabeled{t,1}+(subTaskdata.n_l);
            indlb = subTaskdata.index_set_Labeled{t,1};
            ind   = [indlb,indUn];
        case 3
            indUn = subTaskdata.index_set_UnLabeled{t,1}+(subTaskdata.n_l);
            indlb = subTaskdata.index_set_Labeled{t,1};
            indQ  = indUn+subTaskdata.n_u;
            ind   = [indlb,indUn,indQ];
    end
end
%% train R,beta,eta, simultaneiously
function [R,beta_t,eta_t,alpha_t]=OptimizeR_beta_eta(n_l,n_u,qc,K_tt,Yl,c_t,d_t,fixedA,AMatrix)
    % in this function A means R 

    %% Initializing Q: the subset of unlabeled data to query from 
        % Assuming Q = U, i.e, Using all unlabeled data for querying, instead
        % of using only a part of unlabeled data searching for query points
        n_q = n_u; % consider all of unlabeled data for selecting query 
        lambda=1; %For now but it must be specified by the caller function 
        if nargin == 7 
            c_t=1;
            d_t=-1;
            fixedR=false;
        end
    %% Initialize some constants    
        ONEA = ones(n_l+n_u+n_q,1);
        ONEL = ones(n_l,1);
        ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
        ONEU = ones(n_u,1);
        ONEQ = ones(n_q,1);
        ONEQMATRIX = ONEQ*ONEQ';
        r_q=[ONEL;1-qc;qc];
        p_q=[zeros(n_l+n_u,1);qc];
    %% Prepare Kernel Matrix 

    if ~fixedA 
            n=n_l+n_u+n_q;
            %% commented:The first form of the problem
    %         cvx_begin sdp
    %             variable   A_uu(n_u,n_u)
    %             variable   A_uq(n_u,n_q)
    %             variable   A_lu(n_l,n_u)
    %             variable   beta_Var(n_l+n_u+n_q,1)
    %             variable   eta_Var(n_l+n_u+n_q,1)
    %             variable   tVar
    %             expression A
    %             % This is the lable matrix which we want to estimate it's
    %             % unlabeled part
    %             A = [Yl*Yl',A_lu,Yl*ONEQ';A_lu',A_uu,A_uq;ONEQ*Yl',A_uq',ONEQMATRIX];
    % 
    %             minimize (tVar)
    %             subject to 
    %                A>=0
    %                diag(A_uu)==ONEU
    %                eta_Var>=0
    %                beta_Var>=0 
    %                [K_tt .* A,ONED-beta_Var+eta_Var;(ONED-beta_Var+eta_Var)',2*(c_t*tVar-beta_Var'*r_q-eta_Var'*p_q-d_t)/lambda ]>=0
    %         cvx_end 
    %        dualval=cvx_optval;
    %        KA=K_tt .* A;
            %% commented:Another form of the above problem: proper for standard solvers
            % it is the same as the next one except having variable beta
    %         M=zeros(n+1,n+1);
    %         M(n+1,n+1)=lambda/2;
    %         E_nn=zeros(n+1,n+1);
    %         E_nn(n+1,n+1)=1;
    %         %M(n+1,1:n)=-r_q/2;
    %         %M(1:n,n+1)=-r_q;
    %         ind=[1:n_l,n_l+n_u+1:n];
    %         ym=ones(n,1);
    %         ym(1:n_l)=Yl;
    %         % E_I,e_nplus
    %         E_I=zeros(n,n+1);
    %         E_I(1:n,1:n)=eye(n);
    %         e_nplus=zeros(n+1,1);
    %         e_nplus(n+1)=1;
    %         cvx_clear
    %         cvx_begin sdp
    %             variable L(n+1,n+1) symmetric 
    %             variable R(n,n)
    %             variable eta_t(n,1)
    %             variable beta_t(n,1)
    %             dual variable zl
    %             minimize ( trace(M'*L)+eta_t'*p_q+beta_t'*r_q )
    %             subject to 
    %                 L==semidefinite(n+1)
    %                 R==semidefinite(n)
    %                 eta_t>=0
    %                 beta_t>=0
    %                 trace(L*E_nn)>=0
    %                 E_I*L*e_nplus==ONED+eta_t-beta_t
    %                 for i=ind
    %                     for j=ind
    %                         R(i,j)==ym(i)*ym(j)
    %                     end
    %                 end
    %                 for i=n_l+1:n_l+n_u
    %                     R(i,i)==1
    %                 end
    %                 for i=1:n
    %                     for j=1:n
    %                         L(i,j)==K_tt(i,j)*R(i,j)
    %                     end
    %                 end
    %         cvx_end
            %% Another form of the above problem: ommitting beta_t var
            M=zeros(n+1,n+1);
            M(n+1,n+1)=lambda/2;
            %M(n+1,1:n)=-r_q/2;
            M(1:n,n+1)=-r_q;
            M=(M+M')/2;

            ind=[1:n_l,n_l+n_u+1:n];
            ym=zeros(n,1);
            ym(1:n_l)=Yl;
            ym(n_l+n_u+1:n)=ones(n_q,1);
            %% commented: Calling CVX    
            % E_I,e_nplus
    %         E_I=zeros(n,n+1);       E_I(1:n,1:n)=eye(n);
    %         e_nplus=zeros(n+1,1);   e_nplus(n+1)=1;
    %         cvx_clear
    %         
    %         cvx_begin sdp
    %             variable L(n+1,n+1) symmetric 
    %             variable R(n,n) symmetric
    %             variable eta_t(n,1)
    %             variable beta_t(n,1)
    %             %+ONED'*r_q
    %             dual variable ale
    %             dual variable alb
    %             minimize ( trace(M'*L)+eta_t'*(p_q+r_q) )
    %             subject to 
    %                 L==semidefinite(n+1)
    %                 R==semidefinite(n)
    %                 ale:eta_t>=0
    %                 alb:beta_t>=0
    %                 %norm(R,'fro')<=n%+stp*square_pos(norm(R,'fro'))
    %                 diag(R) <=1
    %                 beta_t==ONED+eta_t-E_I*L*e_nplus
    %                 for i=ind
    %                     for j=ind
    %                         R(i,j)==ym(i)*ym(j)
    %                     end
    %                 end
    %                 for i=n_l+1:n_l+n_u
    %                     R(i,i)==1
    %                 end
    %                 for i=1:n
    %                     for j=1:n
    %                         L(i,j)==K_tt(i,j)*R(i,j)
    %                     end
    %                 end
    %         cvx_end
    %         dualval = cvx_optval;
    %         beta_t=ONED+eta_t-E_I*L*e_nplus;
            %% Using MOSEK to optimize   
            [optobjval,R,beta_t,eta_t,alpha_t]=mosekOptRbetaeta(n_l,n_u,n_q,p_q,r_q,M,ym,K_tt,ONED,c_t);
            %% commented:Another form of the above problem: ommitting R if possible
            % ommitting R results to a non-equivalent problem. 
            % Can we add spread constraints on eigenvalue of K \odot R and get
            % the same result? No, it gets different result. 
    %         ev=eig(K_tt);
    %         sev=min(ev);
    %         lev=max(ev);
    %         M=zeros(n+1,n+1);
    %         M(n+1,n+1)=lambda/2;
    %         E_nn=zeros(n+1,n+1);
    %         E_nn(n+1,n+1)=1;
    %         %M(n+1,1:n)=-r_q/2;
    %         M(1:n,n+1)=-r_q;
    %         ind=[1:n_l,n_l+n_u+1:n];
    %         ym=ones(n,1);
    %         ym(1:n_l)=Yl;
    %         % E_I,e_nplus
    %         E_I=zeros(n,n+1);
    %         E_I(1:n,1:n)=eye(n);
    %         e_nplus=zeros(n+1,1);
    %         e_nplus(n+1)=1;
    %         DM=zeros(n,n);
    %         for i=ind
    %             for j=ind
    %                 DM(i,j)=K_tt(i,j)*ym(i)*ym(j);
    %             end
    %         end
    %         for i=n_l+1:n_l+n_u
    %             DM(i,i)=K_tt(i,i);
    %         end
            %% commented:It doesnot worked correctly, till now. 
    %         cvx_clear
    %         
    %         cvx_begin sdp
    %             variable L(n+1,n+1) symmetric 
    %             variable R(n,n) symmetric
    %             variable eta_t(n,1)
    %             
    %             dual variable zl
    %             minimize ( trace(M'*L)+eta_t'*(p_q+r_q)+ONED'*r_q )
    %             subject to 
    %                 L==semidefinite(n+1)
    %                 %L(1:n,1:n)./K_tt==semidefinite(n)
    %                 L(1:n,1:n)>=sev*eye(n)
    %                 L(1:n,1:n)<=lev*eye(n)
    %                 eta_t>=0
    %                 trace(L*E_nn)>=0
    %                 E_I*L*e_nplus<=ONED+eta_t
    %                 for i=ind
    %                     for j=ind
    %                         L(i,j)==K_tt(i,j)*ym(i)*ym(j)
    %                     end
    %                 end
    %                 for i=n_l+1:n_l+n_u
    %                     L(i,i)==K_tt(i,i)
    %                 end
    % %                 for i=1:n
    % %                     for j=1:n
    % %                         L(i,j)==K_tt(i,j)*R(i,j)
    % %                     end
    % %                 end
    %         cvx_end
            %% commented: Another form of the problem
    %         cvx_begin sdp
    %             variable   R(n,n)
    %             
    %             variable   beta_Var(n_l+n_u+n_q,1)
    %             variable   eta_Var(n_l+n_u+n_q,1)
    %             variable   tVar
    %             
    %             % This is the lable matrix which we want to estimate it's
    %             % unlabeled part
    %             %A = [Yl*Yl',A_lu,Yl*ONEQ';A_lu',A_uu,A_uq;ONEQ*Yl',A_uq',ONEQMATRIX];
    % 
    %             minimize (tVar)
    %             subject to 
    %                R>=0
    %                eta_Var>=0
    %                beta_Var>=0 
    %                [K_tt .* R,ONED-beta_Var+eta_Var;(ONED-beta_Var+eta_Var)',2*(c_t*tVar-beta_Var'*r_q-eta_Var'*p_q-d_t)/lambda ]>=0
    %                for i=ind
    %                     for j=ind
    %                         R(i,j)==ym(i)*ym(j)
    %                     end
    %                end
    %                for i=n_l+1:n_l+n_u
    %                     R(i,i)==1
    %                end
    %         cvx_end 
            %% computing alpha: 
            %c_t=1/((subTaskdata.T-1)*muPar+lambda_t);
    %         KA=K_tt.*R;
    %         pKyinv=pinv(KA);
    %         n_q=n_u;
    %         ONED = [ones(n_l+n_u,1);zeros(n_q,1)];
    %         alpha_t=pKyinv*(ONED+eta_t-beta_t)/c_t;
            %% checking strong duality
    %         prvalusingdual=-0.5*c_t* alpha_t'*KA*alpha_t+(ONED)'*alpha_t;
    %         cvx_clear 
    %         
    %         cvx_begin
    %             variable alpha_d(n_l+n_u+n_q,1)
    %             dual variable er
    %             dual variable br
    %             
    %             maximize (-0.5*c_t* alpha_d'*KA*alpha_d+(ONED)'*alpha_d+d_t)
    %             subject to 
    %                 er:alpha_d>=-p_q
    %                 br:alpha_d<=r_q
    %         cvx_end
    %         alpha_t=alpha_d;
    %         primalval=cvx_optval;
    %         dualval-primalval

    else % it is a socp program but for now it is coded as a sdp
            cvx_begin sdp
                variable   beta_Var(n_l+n_u+n_q,1)
                variable   eta_Var(n_l+n_u+n_q,1)
                variable   tVar
                % This is the lable matrix which we want to estimate it's
                % unlabeled part
                minimize (tVar)
                subject to 
                   eta_Var>=0
                   beta_Var>=0 
                   [K_tt .* AMatrix,ONED-beta_Var+eta_Var;(ONED-beta_Var+eta_Var)',(c_t*tVar-beta_Var'*r_q-eta_Var'*p_q-d_t)/lambda ]>=0
            cvx_end     
            R=AMatrix; 
    end
end
%% Update query st_ep in alternating minimization of AL for all tasks
function [q,A]=UpdateQueryMulti_tasks()

end
%% Update query step in alternating minimization of Active Learning for a single subTask
function [q,A]=UpdateQuery(n_l,n_u,Kll,Klu,Kuu,eta_Varc,beta_Varc)

    %% Use all of unlabeled data for selecting query
    n_q = n_u; % consider all of unlabeled data for selecting query
    Klq=Klu;Kqu=Kuu;Kqq=Kuu;% Use all of unlabeled data for query
    bSize  = 2;
    g = 0; % For now! 
%% Initialize some constants    
    ONEA = ones(n_l+n_u+n_q,1);
    ONEL = ones(n_l,1);
    ONED = [ones(n_l+n_u,1);zeros(n_q,1)];    
    ONEU = ones(n_u,1);
    ONEQ = ones(n_q,1);
    ONEQMATRIX = ONEQ*ONEQ';
   
    cvx_begin sdp
        variable   A_uu(n_u,n_u)% lable matrix for unlabeled data
        variable   A_uq(n_u,n_q)% lable matrix for unlabeled and query data
        variable   A_lu(n_l,n_u)% lable matrix for labeled and query data
        variable   q(n_q,1)     % query variable, the largest elements are for querying
        variable   tVar         % objective function
        expression K            % Kernel Matrix as a whole 
        expression A            % Lable Matrix as a whole  
        expression ASmall1      
        expression ASmall2
        expression r_q          
        expression p_q

        r_q=[ONEL;1-q;q];
        p_q=[zeros(n_l+n_u,1);q];
        K = [Kll,Klu,Klq;Klu',Kuu,Kuq;Klq',Kuq',Kqq];
        % This is the lable matrix which we want to estimate it's
        % unlabeled part
        A = [Yl*Yl',A_lu,Yl*ONEQ';A_lu',A_uu,A_uq;ONEQ*Yl',A_uq',ONEQMATRIX];
        ASmall1 = [Yl*Yl',A_lu;A_lu',A_uu];
        ASmall2 = [A_uu,A_uq';A_uq,ONEQMATRIX];
        % Declare Optimization problem for CVX
        minimize (tVar)
        subject to
           0 <= q <=1;
           %A>=0;
           ASmall1>=0;
           ASmall2>=0;
           q'*ONEQ==bSize;
           diag(A_uu)==ONEU;
           [K .* A,ONED-beta_Varc+eta_Varc;(ONED-beta_Varc+eta_Varc)',2*(tVar-beta_Varc'*r_q-eta_Varc'*p_q)/lambda ]>=0;
    cvx_end  
end
%% Mosek Optimize R,beta,eta
function [optobjval,R1,beta_t,eta_t,alpha_t]=mosekOptRbetaeta(n_l,n_u,n_q,p_q,r_q,M,ym,K_tt,d_Y,c_t)

        n = n_l+n_u+n_q;
        ONED = [ones(n_l+n_u,1);zeros(n_q,1)];
        %ONED = [ones(n_l+n_u,1);zeros(n_q,1)]; 
        
        [r, res] = mosekopt('symbcon');
        % Matrix variables(semidefinite)
        % L : variable number 1 
        % R : variable number 2 
        % vector variable : eta_t
        
        % defining Objective function : <M,L>+eta_t'*(p_q+r_q)
        prob.c         = p_q+r_q;
        prob.bardim    = [n+1,n];
        % defining matrix M in <M,L>
        prob.barc.subj = ones(1,n+1);       % the first semdefinite variable L, coefficient
        prob.barc.subk = (n+1)*ones(1,n+1); % row    n+1
        prob.barc.subl = 1:(n+1);           % column 1:n
        prob.barc.val  = [M(:,n+1)];%-r_q/2;lambda/2];
        % Defining upper and lower variables bound(vector variables)
        prob.blx         = zeros(1,n);
        prob.bux         = [];
        % Defining Constraints
        % constraints : 1_D+eta-E_I*L*e_{n+} >=0 or -ONED<=eta_t-E_I*L*e_nplus
            % lower and upper bound for constraints 
        prob.blc       = -d_Y';
        prob.buc       = inf(1,n);
        
        constr_begin   = 0;
        constr_end     = n;
        constr_num     = constr_begin+1:constr_end;
        constr_numa    = constr_num;
    
        prob.bara.subi = constr_num;            % constraints constr_begin+1:constr_end 
        prob.bara.subj = ones(1,n);             % Variable number 1 (L) multiplier C_j: <L,C_j> 
        prob.bara.subk = (n+1)*ones(1,n);       % Last Row  C_j 
        prob.bara.subl = 1:n;                   % Column 1 to n of C_j
        prob.bara.val  = -ones(1,n)/2;          % -E_I*L*e_{n+1}
        dualconstrnum  = constr_num;
        % defining constraints : R_ij = y_i*y_j
        n_cR   = (n_l+n_q)*(n_l+n_q+1)/2;
        idx_lq = [1:n_l,n_l+n_u+1:n];
        Ain    = zeros(n);
        Ain(idx_lq,idx_lq)=bsxfun(@ge, (idx_lq).', idx_lq);
        [indr,indc]    = find(Ain);indr   = indr';indc   = indc';
        Aym   = ym*ym';
        sAym  = 2*Aym-eye(n);
        IND   = sub2ind(size(Aym),indr,indc);
        prob.blc       = [prob.blc,sAym(IND)];
        prob.buc       = [prob.buc,sAym(IND)];
        constr_begin   = constr_end;
        constr_end     = constr_end + n_cR;       
        constr_num     = constr_begin+1:constr_end;

        prob.bara.subi = [prob.bara.subi,constr_num];
        prob.bara.subj = [prob.bara.subj,2*ones(1,n_cR)]; % 2 is R Matrix variable 
        prob.bara.subk = [prob.bara.subk,indr];
        prob.bara.subl = [prob.bara.subl,indc];
        prob.bara.val  = [prob.bara.val,ones(1,n_cR)]; 
        MAt= [prob.bara.subi;prob.blc;prob.buc;prob.bara.subk;prob.bara.subl;prob.bara.val]
        % defining constraints : R_ii = 1 , i\epsilon D_u
        n_cRii  = n_u;
        constr_begin   = constr_end;
        constr_end     = constr_end + n_cRii;       
        constr_num     = constr_begin+1:constr_end;
        indr   = n_l+1:n_l+n_u;
        indc   = indr;
        prob.blc       = [prob.blc,ones(1,n_cRii)];
        prob.buc       = [prob.buc,ones(1,n_cRii)];
        prob.bara.subi = [prob.bara.subi,constr_num];
        prob.bara.subj = [prob.bara.subj,2*ones(1,n_cRii)]; % 2 is R Matrix variable 
        prob.bara.subk = [prob.bara.subk,indr];
        prob.bara.subl = [prob.bara.subl,indc];
        prob.bara.val  = [prob.bara.val ,ones(1,n_cRii)];
        
        % defining constraints : L_ij= R_ij*K_ij
        n_cL   = n*(n+1)/2;
        idx_all=1:n;
        Ain    =zeros(n);
        Ain(idx_all,idx_all) = bsxfun(@ge,(idx_all).',idx_all);
        [indr,indc]   = find(Ain);indr=indr';indc=indc';
        
        constr_begin   = constr_end;
        constr_end     = constr_end + n_cL;       
        constr_num     = constr_begin+1:constr_end;
        

        prob.blc       = [prob.blc,zeros(1,n_cL)];
        prob.buc       = [prob.buc,zeros(1,n_cL)];
        % variable L
        prob.bara.subi = [prob.bara.subi,constr_num];
        prob.bara.subj = [prob.bara.subj,ones(1,n_cL)]; % variable 1:L
        prob.bara.subk = [prob.bara.subk,indr];
        prob.bara.subl = [prob.bara.subl,indc];
        Aym   = ones(n);
        IND   = sub2ind(size(Aym),indr,indc);
        prob.bara.val  = [prob.bara.val,Aym(IND)];
        
        % variable R
        prob.bara.subi = [prob.bara.subi,constr_num];       % the same constraints
        prob.bara.subj = [prob.bara.subj,2*ones(1,n_cL)];   % variable 2:R
        prob.bara.subk = [prob.bara.subk,indr];             % the same indices
        prob.bara.subl = [prob.bara.subl,indc];             % "
        Aym   = -K_tt;                                      % L_ij-K_ij*R_ij=0
        prob.bara.val  = [prob.bara.val,Aym(IND)];
        coeffval = ones(1,n);
        prob.a         = sparse(constr_numa,(1:n),coeffval,constr_end,n);
        

        %% Solve the problem using MOSEK
        [r,res]        = mosekopt('minimize info',prob); 
        %% Retrieve and compare the result
        optobjval=res.sol.itr.pobjval;
        nL  = (n+2)*(n+1)/2;
        idx_all=1:(n+1);
        Ain    =zeros(n+1);
        Ain(idx_all,idx_all) = bsxfun(@ge,(idx_all).',idx_all);
        [indr,indc]   = find(Ain);indr=indr';indc=indc';
        IND   = sub2ind(size(Ain),indr,indc);
        L1=zeros(n+1);
        L1(IND) = res.sol.itr.barx(1:nL);
        L1=L1+tril(L1,-1)';
        eta_t = res.sol.itr.xx; 
        mu_t  = res.sol.itr.slx;
        idx_all=1:(n);
        Ain    =zeros(n);
        Ain(idx_all,idx_all) = bsxfun(@ge,(idx_all).',idx_all);
        [indr,indc]   = find(Ain);indr=indr';indc=indc';
        IND   = sub2ind(size(Ain),indr,indc);
        R1=zeros(n);
        R1(IND)=res.sol.itr.barx(nL+1:end);
        R1 = R1+tril(R1,-1)';
        beta_t=d_Y+eta_t-L1(1:n,n+1);
        nu_t  = res.sol.itr.slc(dualconstrnum);
        %% some codes for checking 
        %                   result of MOSEK with CVX
        %                   and Alpha_t with dual using strong duality
%         KA=K_tt.*R1;
%         pKyinv=pinv(KA);
%         n_q=n_u;
%         ind=[1:n_l,n_l+n_u+1:n];
% 
%         cvx_begin sdp
%                 variable L(n+1,n+1) symmetric 
%                 variable R(n,n) symmetric
%                 variable eta_t(n,1)
%                 variable beta_t(n,1)
%                 %+ONED'*r_q
%                 dual variable mu_t1
%                 dual variable nu_t1
%                 minimize ( trace(M'*L)+eta_t'*(p_q+r_q) )
%                 subject to 
%                     L==semidefinite(n+1)
%                     R==semidefinite(n)
%                     mu_t1:eta_t>=0
%                     nu_t1:beta_t>=0
%                     %norm(R,'fro')<=n%+stp*square_pos(norm(R,'fro'))
%                     diag(R) <=1
%                     beta_t==ONED+eta_t-L(1:n,n+1);
%                     for i=ind
%                         for j=ind
%                             R(i,j)==ym(i)*ym(j)
%                         end
%                     end
%                     for i=n_l+1:n_l+n_u
%                         R(i,i)==1
%                     end
%                     for i=1:n
%                         for j=1:n
%                             L(i,j)==K_tt(i,j)*R(i,j)
%                         end
%                     end
%             cvx_end       
% %         prvalusingdual=-0.5*c_t* alpha_t'*KA*alpha_t+(d_Y)'*alpha_t;
%         cvx_clear 
%         cvx_begin
%             variable alpha_t(n_l+n_u+n_q,1)
%             dual variable er
%             dual variable br
%             
%             maximize (-0.5*c_t* alpha_t'*KA*alpha_t+(d_Y)'*alpha_t)
%             subject to 
%                 er:alpha_t>=-p_q
%                 br:alpha_t<=r_q
%         cvx_end
%         abs(mu_t-mu_t1)
%         abs(nu_t-nu_t1)
        %% computing alpha_t
        alpha_t=compalpha(beta_t,nu_t,eta_t,mu_t,r_q,p_q);
         
%         norm(alpha_t-alpha_t1)/norm(alpha_t)
end