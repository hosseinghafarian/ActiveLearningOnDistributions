function [ALresult,model_set,Y_set]= ActiveDConvexRelaxOutlierProxADMM2(WMST_beforeStart, Model, learningparams)
        
        global      cnstData
        ALresult.active  = true;
        initL     = cnstData.initL;
        unlabeled = cnstData.unlabeled;
        n_l       = cnstData.n_l;
        n_u       = cnstData.n_u;
        batchSize = cnstData.batchSize;
        n_o       = cnstData.n_o;
        Kll       = cnstData.K(initL',initL');
        Kuu       = cnstData.K(unlabeled',unlabeled');
        Klu       = cnstData.K(initL',unlabeled');
        lambda    = learningparams.lambda;
        lambda_o  = learningparams.lambda_o;
        n_o       = 4;
        % code with WARMSTART 
        n         = n_l+n_u;     % size of data 
        c         = 1;           % parameter for finding absolute value of y_u.*(1-q) in cost function 
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
        
        
        %% Define Constraints and Objective function 
         e_nplus1 = [ones(nap+1,1);0];
        % the following matrix are for making semidefinite matrix of the
        % original problem,i.e. KV= CMat.* X, GPlus = DMat.* X
        DMat  = [ones(nap+1,nap+1),[zeros(nap,1);0];zeros(1,nap+2)];% e_nplus1*e_nplus1';
        CMat  = [K,zeros(nap,1),ones(nap,1);zeros(1,nap),0,0;ones(1,nap),0,1];
        epsconst = 0.00001;
        %% Initialization
        
        q     = rand(n_q,1);   
        r     = rand(n_u,1);%_u,1);
        p     = rand(n  ,1);
        w_o    = rand(n  ,1);
        s      = rand(nap,1);
        v     = rand(n_u,1);

        h_s    = rand(nap ,1);
        h_d    = rand(nap ,1);
        
        sdh    = h_d;
        x_s    = rand(nap ,1);
        x_d    = rand(nap ,1);
        sdx    = x_d;
        z_s    = rand(nap ,1);
        z_d    = rand(nap ,1);
        sdz    = z_d;
        close all
        for i=1:10
                hfig(i)=figure;
                
                prism;
            end
        [X_lower,...
                h_d,x_d,z_d,...
                h_s,x_s,z_s,...
                diffonprojX] = initializeX(n_q,n_l,n,Yl,CMat);
            
        rhoAUGLG = 8;
        multirho = 1.005;
        close all;
        converged = false;
        repeati = 1
        while ~converged
            %% Save Previous data
            qpre   = q;
            ppre   = p;
            rpre   = r;
            spre   = s;
            w_opre = w_o;
            vpre   = v;
            X_lowerpre = X_lower;
            
            %% Subproblem 1: SDP problem    
            [X_lower,h_d,x_d,z_d,diffonprojX]       =...
                        ADMM_LEVENMARQUADT_X(lambda,n,n_l,n_u,nap,X_lower ,CMat,...
                                                h_s-sdh,x_s-sdx,z_s-sdz,rhoAUGLG);
            %% Subproblem 2
            [h_s,x_s,z_s,s,w_o,p,q,v,r,rl] = ADMM_qrpu(n_q,n_l,n,n_o,lambda_o,batchSize,Yl,K,K_l,...
                                                    h_d+sdh,x_d+sdx,z_d+sdz,...
                                                    s,w_o,p,q,v,r,rhoAUGLG,c);
                                           
            %% LAGRANGE UPDATING STEP
            sdh  = sdh+ h_d-h_s;
            sdx  = sdx+ x_d-x_s;
            sdz  = sdx+ z_d-z_s;
            
            %% Showing results
            sdres(repeati,1) = norm(s  -spre);
            sdres(repeati,2) = norm(r  -rpre);
            sdres(repeati,3) = norm(q  -qpre);
            sdres(repeati,4) = norm(w_o-w_opre);
            sdres(repeati,5) = norm(v  -vpre);
            sdres(repeati,6) = norm(h_d-h_s);
            sdres(repeati,7) = norm(x_d-x_s);
            sdres(repeati,8) = norm(z_d-z_s);
            sdres(repeati,9) = norm(p  -ppre);%X_lowerpre - X_lower);
            norm(s  -spre)
            norm(r  -rpre)
            norm(q  -qpre)
            norm(w_o-w_opre)
            norm(v  -vpre)
            norm(h_d-h_s)
            norm(x_d-x_s)
            norm(z_d-z_s)
            norm(p  -ppre)
            %sdres(repeati,10)= diffonprojX;
            for i=1:9
                figure(hfig(i));
                plot (sdres(1:repeati,i));
                prism;
            end
            
            %% Rest 
            rhoAUGLG = rhoAUGLG *multirho;
            c = rhoAUGLG;
            df = norm(w_o-w_opre)+norm(s  -spre)+norm(q  -qpre) %norm(X_lowerpre - X_lower);
            if df <epsconst
                converged=true;
            end
            repeati = repeati +1
        end
end
function [X_lower,...
                h_d,x_d,z_d,...
                h_s,x_s,z_s,...
                diffonprojX] = initializeX(n_q,n_l,n,Yl,CMat)
        nap    = n+n_q;
        
        X_yl   = Yl*Yl';
        X_lower= rand((nap+2)*(nap+3)/2,1);
        
        % expand to full size matrix
        nx  = nap + 2;
        b   = tril(ones(nx));
        b(b == 1) = X_lower;
        
        X   = b;
        X   = (X + X');
        X   = X - diag(diag(X)) / 2;
        X ( 1:n_l,1:n_l ) = X_yl;
        X ( nap+1,nap+1 ) = 1;
        
        X1 = proj_ontwo_cones(nap,X,CMat);
        diffonprojX = norm(X1-X,'fro');
        X = X1;
        z_d = ones(nap,1);%diag(X(1:nap,1:nap));
        z_s = z_d;
        x_d = 0.5*ones(nap,1);%X(1:nap,nap+1);
        x_s = x_d;
        h_d = 0.5*ones(nap,1);%X(1:nap,nap+2);
        h_s = h_d;
        
        X(eye(nap+2) == 1) = X(eye(nap+2) == 1) ./ sqrt(2);
        
        X_lower = X(tril(ones(nap+2)) == 1);
end
function X         = proj_ontwo_cones(nap,X,CMat)
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
norm(X([1:nap,nap+2],[1:nap,nap+2]).*CMat([1:nap,nap+2],[1:nap,nap+2]) -XCMat,'fro')
X   = (X + X')/2;

%X = XL;
[V,S] = eig(X(1:nap+1,1:nap+1));
S = diag(S);
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
function [ varib ]  = projrange(l,u,varib)
% if u is less than l in some elements, project them first
isbel  = u < l;
u(isbel)= l(isbel);

isbel  = varib < l;
varib (isbel) = l(isbel);
isabo  = varib >= u;
varib (isabo) = u(isabo);
end

%% Subproblem 2
function [h_s,x_s,z_s,s,w_o,p,q,v,r,rl] = ADMM_qrpu(n_q,n_l,n,n_o,lambda_o,bSize,Yl,K,K_l,...
                                                    h,x,z,...
                                                    s,w_o,p,q,v,r,rhoAUGLG,c)
                                                
ONED   = [ones(n,1);zeros(n_q,1)];
n_u    = n_q; nap = n+n_q;%% assume all of data is used and n_u==n_q
w_ovar = sdpvar(n,1);
pvar   = sdpvar(n,1);
qvar   = sdpvar(n_q,1);
vvar   = sdpvar(n_u,1);
svar   = sdpvar(n+n_q,1);
rvar   = sdpvar(n_u,1);
g_D    = sdpvar(n+n_q,1);
hvar   = sdpvar(n+n_q,1);
xvar   = sdpvar(n+n_q,1);
zvar   = sdpvar(n+n_q,1);

cConstraint = [-pvar<=K(1:n,1:n)*w_ovar<=pvar,pvar<=1, sum(pvar) == n_o];
cConstraint = [cConstraint,pvar>=0,qvar>=0,rvar>=0,sum(qvar)==bSize];
cConstraint = [cConstraint,rvar+pvar(n_l+1:n,1)+qvar==1];
cConstraint = [cConstraint,g_D(1:n_l)==0,g_D(n_l+1:n)==1-rvar,g_D(n+1:n+n_q)==0];
cConstraint = [cConstraint,[Yl-K_l*w_ovar;vvar;qvar]==xvar,ONED-g_D-svar==hvar];
cConstraint = [cConstraint,rvar           >=zvar(n_l+1:n)];
cConstraint = [cConstraint,1-pvar(n_l+1:n)>=zvar(n_l+1:n)];
cConstraint = [cConstraint,1-qvar         >=zvar(n_l+1:n)];
cConstraint = [cConstraint,1-pvar(1:n_l)  >=zvar(1:n_l),...
                                      qvar>=zvar(n+1:nap),zvar>=0];
cConstraint = [cConstraint,-rvar<=vvar<=rvar];
%cConstraint = [cConstraint,

cObjective  = lambda_o/2*norm(w_ovar)^2+c*sum(rvar)+...
                    norm(svar(n+1:n+n_l),1)+ sum(max(svar(1:n),0))+...
                    rhoAUGLG/2* norm([hvar;xvar;zvar]-[h;x;z],2)^2;
sol = optimize(cConstraint,cObjective);
if sol.problem == 0
   w_o  = value(w_ovar);
   p    = value(pvar);
   rl   = value(Yl-K_l*w_ovar);
   r    = value(rvar);
   q    = value(qvar);
   v    = value(vvar);
   s    = value(svar);
   h_s  = value(hvar);
   x_s  = value(xvar);
   z_s  = value(zvar);
else
    assert(true,'cannot solve for w_o,p');
end                                        
                                                
end
            %% Subproblem 1: SDP problem    
function   [X_lower,h_d,x_d,z_d,diffonprojX]       =...
                        ADMM_LEVENMARQUADT_X(lambda,n,n_l,n_u,nap,X_lower ,CMat,...
                                                h,x,z,rhoAUGLG)
nx = nap + 1;n_q = n_u; % assume all of u is used for query 
epsd = 0;
% bs = tril(ones(nx));
% bs(bs == 1) = X_lower;
% X = bs;
% X = (X + X');
% X = X - diag(diag(X)) / 2;
%% in projection method we have to use also project to domain
% idx    = h>1;
% h(idx) = 1-epsd;
% idx    = h<-1;
% h(idx) = -1;
% 
% idx    = z>=1;
% z(idx) = 1-epsd;
% idx    = z<=0;
% z(idx) = epsd;
% lb     = [-ones(n,1)*(1-epsd);epsd*ones(n_q,1)];%zeros(n_q,1)];
% ub     = ones(nap,1)*(1-epsd);
% idx    = x<lb;
% x(idx) = lb(idx);
% idx    = x>ub;
% x(idx) = ub(idx);
%% store X parts 
% %[X] = storeXparts(n,n_l,nap,n_u,X,Yl,w_o+sdw_o,K_l,q+sdq,s+sds,ravg-sdr_X,t+sdt,pavg-sdp_X,dX_c+sdX_X);
% X(1:nap,nap+1) = x;
% X(nap+1,1:nap) = x';
% 
% X(1:nap,nap+2) = h;
% X(nap+2,1:nap) = h';
% diagX = eye(nap);
% X(eye(nap+2)==1) = [z;diag(X(nap+1:nap+2,nap+1:nap+2))];
% 
% %% Project onto Cones
% X1 = proj_ontwo_cones(nap,X,CMat);
% diffonprojX  = norm(X1-X,'fro');
% X = X1;
L = sdpvar(nap+1,nap+1);
R = sdpvar(nap+1,nap+1);
%cConstraint= [L(1:nap,nap+1)==h,R(1:nap,nap+1)==x];
cConstraint= [L(1:nap,1:nap)==R(1:nap,1:nap).*CMat(1:nap,1:nap)];
%cConstraint= [cConstraint,diag(R)==[z;1]];
cConstraint= [cConstraint,L>=0,R>=0];
cObjective = lambda/2*L(nap+1,nap+1)+rhoAUGLG/2*norm(L(1:nap,nap+1)-h)^2+rhoAUGLG/2*norm(R(1:nap,nap+1)-x)^2+rhoAUGLG/2*norm(diag(R)-[z;1])^2;
sol = optimize(cConstraint,cObjective);
if sol.problem==0
    X1 = value(R); 
    
   %% Extract X parts
    x_d = X1(1:nap,nap+1);
    h_d = value(L(1:nap,nap+1));
    z_d = diag(X1(1:nap,1:nap));
    diffonprojX  = 0;%norm(X1-X,'fro');
    X = X1;
%     idx      = z_d>=1;
%     z_d(idx) = 1-epsd;
%     idx      = z_d<=0;
%     z_d(idx) = epsd;
%     lb     = [-ones(n,1)*(1-epsd);epsd*ones(n_q,1)];%zeros(n_q,1)];
%     ub     = ones(nap,1)*(1-epsd);
%     idx      = x_d<lb;
%     x_d(idx) = lb(idx);
%     idx      = x_d>ub;
%     x_d(idx) = ub(idx);
    X(eye(nap+1) == 1) = X(eye(nap+1) == 1) ./ sqrt(2);
    X_lower = X(tril(ones(nap+1)) == 1);
    return;
else
    assert(sol.problem==0);
end
%% Extract X parts
x_d = X(1:nap,nap+1);

h_d = X(1:nap,nap+2);

z_d = diag(X(1:nap,1:nap));

X(eye(nap+2) == 1) = X(eye(nap+2) == 1) ./ sqrt(2);
X_lower = X(tril(ones(nap+2)) == 1);
% make 1 and zero cells 
%X (nap+1,nap+1) =1;
X (nap+2,nap+1) =0;
X (nap+1,nap+2) =0;
%% Extract Lower Triangle
% scale diags by 1/sqrt(2)

                                                                   
end
            