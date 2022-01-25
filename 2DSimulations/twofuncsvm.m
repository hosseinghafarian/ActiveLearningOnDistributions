function [info]=twofuncsvm(K,type,xapp,yapp,lambda,lambda_o,wlambda,epsilon,kernel,kerneloption,rho,initL,K_o,n_o)

n=size(xapp,1);
options=sdpsettings('verbose',0);
n_oe=2;
ONESn=ones(n,1);
Ky=K.*(yapp*yapp');
%% YALMIP: Solving quadratic Optimization problem using YALMIP 
switch type %optimization of (1-y_i W_o^T x_i-y_i w^T x_i)_+
    case 1 % Two function but addition of them not with multiplication of loss functions of them. 
        outlierind=n;
        valo=zeros(n,1);
        valo(outlierind)=1;
        alpha_r=sdpvar(n,1);
        h=sdpvar(n,1);
        r=sdpvar(n,1);
        g=sdpvar(n,1);
        t=sdpvar(1,1);
        s=sdpvar(n,1);

        s          =alpha_r.*yapp+h-r;
        fnormterm  =sdpvar(1,1);
        w_onormterm=sdpvar(1,1);

        fnormterm  = alpha_r'*Ky*alpha_r;
        w_onormterm= s'*K*s;

        Objective= fnormterm/(2*lambda)+w_onormterm/(2*lambda_o)-ONESn'*alpha_r+ONESn'*g+n_o*t;
        
        Constraints = [alpha_r>=0,alpha_r<= 1,h>=0,r>=0,g>=0,h+r==g+t*ONESn];%contraint about t is removed. 

        sol=optimize(Constraints,Objective,options);
        
        info.solproblem = sol.problem;
        if sol.problem==0 
             info.type  =type;
            info.alpha_r=value(alpha_r);
            info.h = value(h);
            info.r = value(r);
            info.t = value(t);
            info.g = value(g);
            info.objval=value(Objective);
            alphnorm=value(fnormterm);
            hrknorm =value(w_onormterm);
            info.normW = alphnorm/(lambda^2);
            info.normW_o = hrknorm/(lambda_o^2);
            info.SigAlpha= ONESn'*info.alpha_r;
            info.w_oTx_i=(info.alpha_r.*yapp+info.h-info.r)'*K/lambda_o;
            info.wTx_i  =(info.alpha_r.*yapp)'*K/lambda;
            sumpi       =sum(abs(info.w_oTx_i));
            info.xi     =max(ONESn-yapp.*(info.w_oTx_i+info.wTx_i)',0);
            xisum       =sum(info.xi);
            pstar=lambda*info.normW/2+lambda_o*info.normW_o/2+xisum;
            dstar=sum(info.alpha_r)-lambda*info.normW/2-lambda_o*info.normW_o/2-sum(info.g)-n_o*info.t;
            strabs = sprintf('Sum of |w_o^T*x_i:%5.2f vs:%5.2f',sumpi,n_o);
            display(strabs);
        else
            display('Problem is not solved successfully');
        end
        %Solving primal problem directly using YALMIP
        a=1;
        walpha_p =sdpvar(n,1);
        w_ohr_p  =sdpvar(n,1);
        xi       =sdpvar(n,1);
        p        =sdpvar(n,1);
%        gwof     =sdpvar(n,1);
        up       =1;
        down     =-1;
        pObjective =sum(xi)+lambda*walpha_p'*K*walpha_p/2+lambda_o*w_ohr_p'*K*w_ohr_p/2;
        pConstraint=[walpha_p>=down,walpha_p<=up,xi>=0,xi>=1-yapp.*(K*walpha_p+K*w_ohr_p),...
                     -p<=K*w_ohr_p<=p,p<=1,ONESn'*p==n_o]%,K(14,:)*w_ohr_p==0.8,K(15,:)*w_ohr_p==-0.8]%,...
%                      gwof>=K*walpha_p+a*K*w_ohr_p-a,gwof<=a+K*walpha_p-a*K*w_ohr_p,...
%                      gwof<=a-K*walpha_p+a*K*w_ohr_p,gwof>=-a-K*walpha_p-a*K*w_ohr_p,...
%                      gwof'*ONESn>=-a*n_o,gwof'*ONESn<=a*n_o];%,gwof'*ONESn>=-ONESn'*K*walpha_p,gwof'*ONESn<=ONESn'*K*walpha_p
        sol=optimize(pConstraint,pObjective,options);
        if sol.problem==0
            info.primalWalpha=value(walpha_p);
            info.pObjective  = value(pObjective);
            info.primalW_ohr =value(w_ohr_p);
            info.primalxi    = value(xi);
            info.primalp     = value(p);
            info.primalwxT_i =info.primalWalpha'*K;
            info.primalw_oxT_i=info.primalW_ohr'*K;
            info.outliertest.f_o  =info.primalw_oxT_i(14);
            info.outliertest.f    =info.primalwxT_i(14);   
%             info.gwof        =value(gwof);
        else 
            display('cannot solve the problem');
        end
        info.primal=true;
    case 2  %Only a single classifier function (SVM)
        alpha_r=sdpvar(n,1);
        fnormterm=alpha_r'*Ky*alpha_r;
        Objective= fnormterm/(2*lambda)-ONESn'*alpha_r;
        Constraints = [alpha_r>=0,alpha_r<= 1];
        sol=optimize(Constraints,Objective,options);
        info.solproblem = sol.problem;
        if sol.problem==0 
            info.type  =type;
            info.alpha_r=value(alpha_r);
            info.objval=value(Objective);
            alphnorm=value(fnormterm);
            info.normW = alphnorm/lambda;
            info.SigAlpha= ONESn'*info.alpha_r;
            info.wTx_i  =(info.alpha_r.*yapp)'*K/lambda;
        else
            display('Problem is not solved successfully');
        end
        info.primal=false;
    case 3  % Learning et*w_o_Txi using both alpha_r and et as convex relaxation
            eup = 2;
            eld = 1; 
            alpha_r=sdpvar(n,1);
            xe = sdpvar(n,1);
            h=sdpvar(n,1);
            r=sdpvar(n,1);
            g=sdpvar(n,1);
            t=sdpvar(1,1);
            s=sdpvar(n,1);
            et=sdpvar(1,1);
            s=xe.*yapp+h-r;
            fnormterm=sdpvar(1,1);
            w_onormterm=sdpvar(1,1);
            %alk=(x+mua*ONESn)'*Ky*(x+mua*ONESn);
            fnormterm=alpha_r'*Ky*alpha_r;
            w_onormterm=s'*K*s;
            
            Objective= fnormterm/(2*lambda)+w_onormterm/(2*lambda_o)-ONESn'*alpha_r+ONESn'*g+n_o*t;
            
            Constraints = [alpha_r>=0,h>=0,r>=0,g>=0,t>=0,alpha_r<=1,h+r==g+t*ONESn,...
                           et>=eld,xe>=0,xe<=eup*alpha_r,xe>= eup*alpha_r+et-eup,xe<=et-eld+eld*alpha_r,xe>=eld*alpha_r];

            sol=optimize(Constraints,Objective,options);
            info.solproblem = sol.problem;
            if sol.problem==0 
                info.type  =type;
                info.alpha_r=value(alpha_r);
                info.h = value(h);
                info.r = value(r);
                info.v = value(t);
                info.g = value(g);
                info.objval=value(Objective);
                alphnorm=value(fnormterm);
                hrknorm =value(w_onormterm);
                info.nu = 0%value(nua);
                info.mu = 0%value(mua);
                info.alphaeta = value(xe);
                info.normW = alphnorm/lambda;
                info.normW_o = hrknorm/lambda_o;
                info.SigAlpha= ONESn'*info.alpha_r;
                info.w_oTx_i=(info.alpha_r.*yapp+info.h-info.r)'*K/lambda_o;
                info.wTx_i  =(info.alpha_r.*yapp)'*K/lambda;
            else
                display('Problem is not solved successfully');
            end            
    case 4
        alpha_r=sdpvar(n,1);
        beta_r = sdpvar(n,1);
        h=sdpvar(n,1);
        r=sdpvar(n,1);
        g=sdpvar(n,1);
        t=sdpvar(1,1);
        s=sdpvar(n,1);
        eta_r=sdpvar(n,1);
        mu_r=sdpvar(n,1);
        nu_r=sdpvar(n,1);
        tempf=sdpvar(n,1);
        tempw=sdpvar(n,1);
        tempwo=sdpvar(n,1);
        tempw =beta_r.*yapp+mu_r-nu_r;
        tempwo=alpha_r.*yapp+r-h;
        tempf=alpha_r.*yapp+eta_r+nu_r-mu_r-s;

        wnormterm=sdpvar(1,1);
        w_o_normterm=sdpvar(1,1);
        fnormterm=sdpvar(1,1);
        
        wnormterm=tempw'*K*tempw;
        wnormtermPre=(beta_r.*yapp)'*K*(beta_r.*yapp);
        w_o_normterm=tempwo'*K*tempwo;
        fnormterm=tempf'*K*tempf;
        
        Objective= wnormterm/(2*wlambda)+w_o_normterm/(2*lambda_o)+fnormterm/(2*lambda)...
                   -ONESn'*alpha_r-ONESn'*beta_r+ONESn'*g+n_o*t+ONESn'*(eta_r+s);


        Constraints = [alpha_r>=0,alpha_r<=1,beta_r>=0,beta_r<=1,h>=0,r>=0,g>=0,s>=0,eta_r>=0,...
                        mu_r>=0,nu_r>=0,h+r+mu_r+nu_r==g+t*ONESn+eta_r+s];%contraint  t>=0 is removed. 


        sol=optimize(Constraints,Objective,options);
        info.solproblem = sol.problem;
        if sol.problem==0 
            info.type  =type;
            info.alpha_r=value(alpha_r);
            info.beta_r =value(beta_r);
            info.h = value(h);
            info.r = value(r);
            info.t = value(t);
            info.g = value(g);
            info.eta_r=value(eta_r);
            info.nu_r =value(nu_r);
            info.mu_r =value(mu_r);
            info.s    =value(s);
            
            info.objval=value(Objective);
            
            info.normW = value(wnormterm)/(wlambda^2);
            info.normW_o = value(w_o_normterm)/(lambda_o^2);
            info.normf = value(fnormterm)/(lambda^2);
            info.SigAlpha= ONESn'*info.alpha_r;
            info.SigBeta = ONESn'*info.beta_r;
            info.woaddterm=(info.r-info.h)'*K;
            info.woalphaterm=(info.alpha_r.*yapp)'*K;
            info.w_oTx_i=(info.alpha_r.*yapp+info.r-info.h)'*K/lambda_o;
            info.wTx_i  =(info.beta_r.*yapp+info.mu_r-info.nu_r)'*K/wlambda;
            info.fTx_i  =(info.alpha_r.*yapp+info.eta_r+info.nu_r-info.mu_r-info.s)'*K/lambda;
            info.allw(1,:)=info.w_oTx_i;
            info.allw(2,:)=info.wTx_i;
            info.allw(3,:)=info.fTx_i;
            sumpi=sum(abs(info.w_oTx_i));
            info.xi=max(ONESn-yapp.*(info.w_oTx_i+info.fTx_i)',0);
            info.zeta=max(ONESn-yapp.*info.wTx_i',0);
            info.sumlossw=sum(info.zeta);
            info.sumlosswof=sum(info.xi);
%             pstar=lambda*info.normW/2+lambda_o*info.normW_o/2+xisum;
%             dstar=sum(info.alpha_r)-lambda*info.normW/2-lambda_o*info.normW_o/2-sum(info.g)-n_o*info.t;
%             strabs = sprintf('Sum of |w_o^T*x_i:%5.2f vs:%5.2f',sabs,n_o);
%             display(strabs);
        else
            display('Problem is not solved successfully');
        end
        falpha_p=sdpvar(n,1);
        walpha_p=sdpvar(n,1);
        w_ohr_p=sdpvar(n,1);
        xi=sdpvar(n,1);
        zeta=sdpvar(n,1);
        p=sdpvar(n,1);
        up=1;
        down=-1;
        pObjective=sum(xi)+sum(zeta)+wlambda*walpha_p'*K*walpha_p/2+lambda*falpha_p'*K*falpha_p/2+...
                +lambda_o*w_ohr_p'*K*w_ohr_p/2;
        pConstraint=[walpha_p>=down,walpha_p<=up,falpha_p>=down,falpha_p<=up,...
                      xi>=0,xi>=1-yapp.*(K*falpha_p+K*w_ohr_p),zeta>=0,zeta>=1-yapp.*(K*walpha_p),...
                     -p<=K*w_ohr_p<=p,p<=1,p'*ONESn==n_o];
        sol=optimize(pConstraint,pObjective,options);
        if sol.problem==0
            info.primalWalpha=value(walpha_p);
            info.pObjective = value(pObjective);
            info.primalW_ohr=value(w_ohr_p);
            info.primalxi = value(xi);
            info.primalp = value(p);
            info.primalfalpha=value(falpha_p);
            info.primalwxT_i =info.primalWalpha'*K;
            info.primalfxT_i =info.primalfalpha'*K;
            info.primalw_oxT_i=info.primalW_ohr'*K; 
            
        end
    case 5% Alternating Optimization of (1-y_i w_o^T x_i) ( 1- y_i w^T x_i)_+
        a=1;
        walpha_p =sdpvar(n,1);
        w_ohr_p  =sdpvar(n,1);
        xi       =sdpvar(n,1);
        zeta     =sdpvar(n,1);
        p        =sdpvar(n,1);
        up       =1;
        down     =-1;
        % alternating optimization
        pzeta    = ones(n,1);
        prePW    = zeros(n,1);
        preWO    = zeros(n,1);
        epsi     = 0.001;
        converged = false;
        while ~converged
            pObjective =sum(xi.*pzeta)+lambda*walpha_p'*K*walpha_p/2;
            pConstraint=[walpha_p>=down,walpha_p<=up,xi>=0,xi>=1-yapp.*(K*walpha_p)];
            sol=optimize(pConstraint,pObjective,options);
            if sol.problem==0
                info.primalWalpha=value(walpha_p);
                info.pObjective  = value(pObjective);
                pxi    = value(xi);
                info.primalwxT_i =(info.primalWalpha)'*K;     
            end   
            pObjective =sum(pxi.*zeta)+lambda_o*w_ohr_p'*K*w_ohr_p/2;
            pConstraint=[zeta>=0,zeta>=1-yapp.*(K*w_ohr_p),-p<=K*w_ohr_p<=p,p<=1,ONESn'*p==n_o];   
            sol=optimize(pConstraint,pObjective,options);
            if sol.problem==0
                info.pObjective  = value(pObjective);
                info.primalW_ohr =value(w_ohr_p);
                pzeta    = value(zeta);
                info.primalp     = value(p);
                info.primalw_oxT_i=(info.primalW_ohr)'*K;     
            end
            orgObjective = sum(pxi.*pzeta)+lambda*info.primalWalpha'*K*info.primalWalpha/2+lambda_o*info.primalW_ohr'*K*info.primalW_ohr/2;
            if norm(info.primalWalpha-prePW)+norm(info.primalW_ohr-preWO) <= epsi
                converged=true;
            end
            prePW=info.primalWalpha;
            preWO=info.primalW_ohr;
        end
        info.type  = type;
        info.primal= true;
    case 6 % Paper: Robutst Support Vector Machine Training via Convex Outlier Ablation
        delt=sdpvar(1,1);
        nur  =sdpvar(n,1);
        omeg=sdpvar(n,1);
        etar =sdpvar(n,1);
        M   =sdpvar(n,n);
        SC  =sdpvar(n+1,n+1);
        G   = Ky;
        
        SC  =[M,etar;etar',1];
        SD  =[G.*M,etar+nur-omeg;(etar+nur-omeg)',2*(delt-omeg'*ONESn+etar'*ONESn)];
        AbObjective = delt;
        AbConstraint=[nur>=0,omeg>=0,etar>=0,etar<=1,SC>=0,SD>=0];
        Absol = optimize(AbConstraint,AbObjective,options);
        if Absol.problem==0
                info.pObjective  = value(AbObjective);
                info.primalW_ohr =value(w_ohr_p);
                pzeta    = value(zeta);
                info.primalp     = value(p);
                info.primalw_oxT_i=(info.primalW_ohr)'*K;     
        end
    case 7 % convex relaxation for supervised (1-y_i w_o^T x_i) ( 1- y_i w^T x_i)_+
        G     =sdpvar(n,n);
        p     =sdpvar(n,1);
        w_o_gy=sdpvar(n,1);
        w_o_g =sdpvar(n,1);
        w_o   =sdpvar(n,1);
        
%       w_o   =yapp.*w_o_gy+w_o_g;
        beta_p=sdpvar(n,1);
        eta_p =sdpvar(n,1);
        t     =sdpvar(1,1);
        YM    = diag(yapp);
        
        GMatrix  = sdpvar(n,n);
        KGMatrix = sdpvar(n,n);
        GMatrix  = [G,1-yapp.*(K*w_o);(1-yapp.*(K*w_o))',1];
        KGMatrix = [YM*K.*G*YM ,1-yapp.*(K*w_o)+eta_p-beta_p;(1-yapp.*(K*w_o)+eta_p-beta_p)',2*t/lambda];
        % Form 1: sum(p)<= n_o
        cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p);
        cConstraint=[beta_p>=0,eta_p>=0,GMatrix>=0,KGMatrix>=0,diag(G)==1-yapp.*(K*w_o),-p<=K*w_o<=p,p<=1,ONESn'*p<=n_o];
        % Form 2: Objective + c*sum(p);
%         cp=0.8;
%         cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p)+cp*sum(p);
%         cConstraint=[beta_p>=0,eta_p>=0,GMatrix>=0,KGMatrix>=0,diag(G)==1-yapp.*(K*w_o),-p<=K*w_o<=p,p<=1];
        sol = optimize(cConstraint,cObjective,options);
        if sol.problem==0
            info.type        =type;
            info.primal      =true;
            info.primalW_ohr =value(w_o);
            info.Gprimal     = value(G);
            info.primalp     = value(p);
            info.primalw_oxT_i=(info.primalW_ohr)'*K;  
            dalpha = dual(cConstraint(1));
            ualpha = dual(cConstraint(2));
            if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
                display('error in computing primal variable alpha in twofuncsvm case 7');
                pause;
                return;
            end
            info.w_oloss     = 1-yapp.*(K*info.primalW_ohr);
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
            info.primalWalpha=value(ualpha); 
            info.pObjective  = value(cObjective);
            info.primalwxT_i =(YM*info.primalWalpha.*info.w_oloss)'*K;
        end
    case 8 % convex relaxation for using H(see our paper) (1-y_i w_o^T x_i) ( 1- y_i w^T x_i)_+
        
        H     =sdpvar(n,n);
        p     =sdpvar(n,1);
        w_o   =sdpvar(n,1);
        
        beta_p=sdpvar(n,1);
        eta_p =sdpvar(n,1);
        t     =sdpvar(1,1);
        YM    = diag(yapp);        
        
        HMatrix  = sdpvar(n+1,n+1);
        KHMatrix = sdpvar(n+1,n+1);
        HMatrix  = [H,yapp-(K*w_o);(yapp-(K*w_o))',1];
        KHMatrix = [K.*H ,1-yapp.*(K*w_o)+eta_p-beta_p;(1-yapp.*(K*w_o)+eta_p-beta_p)',2*t/lambda];
        
        cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p);
        cConstraint=[beta_p>=0,eta_p>=0,HMatrix>=0,KHMatrix>=0,diag(H)==1-yapp.*(K*w_o),-p<=K*w_o<=p,p<=1,ONESn'*p<=n_o];
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            info.type=type;
            info.primal=true;
            info.primalW_ohr =value(w_o);
            info.Hprimal     = value(H);
            info.primalp     = value(p);
            info.primalw_oxT_i=(info.primalW_ohr)'*K;  
            dalpha = dual(cConstraint(1));
            ualpha = dual(cConstraint(2));
            if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
                display('error in computing primal variable alpha in twofuncsvm case 7');
                pause;
                return;
            end
            info.w_oloss     = 1-yapp.*(K*info.primalW_ohr);
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
            info.primalWalpha=value(ualpha); 
            info.pObjective  = value(cObjective);
            info.primalwxT_i =(YM*info.primalWalpha)'*K;
        end
    case 9 % convex relaxation for semisupervised (1-y_i w_o^T x_i) ( 1- y_i w^T x_i)_+
           % based on replacing y_i.* w_o^t \phi(x_i) by p_i  
        H     =sdpvar(n,n);
        p     =sdpvar(n,1);
        w_o   =sdpvar(n,1);
        g     =sdpvar(n,1);
        Pl    =sdpvar(n,n);
        
        %yun   =sdpvar(n-n_l,1);%unlabeled data labels
        n_l   =size(initL,2);
        n_u   =n-n_l;
        X     =sdpvar(n_u,2);
        
        beta_p=sdpvar(n,1);
        eta_p =sdpvar(n,1);
        t     =sdpvar(1,1);
        yun   =sdpvar(n_u,1);
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        yall  =sdpvar(n,1);
        yall(initL)     =yapp(initL);
        yall(setunlab)  =yun;
        yab             =sdpvar(n,1);
                 
        HMatrix      = sdpvar(n+1,n+1);
        KHMatrix     = sdpvar(n+1,n+1);
%         PMatrix      = sdpvar(n+1,n+1);
%         PMatrix      = [Pl,p;p',1];
        HMatrix      = [H,yall-(K*w_o);(yall-(K*w_o))',1];
        KHMatrix = [K.*H ,yapp.*(yall-(K*w_o))+eta_p-beta_p;(yapp.*(yall-(K*w_o))+eta_p-beta_p)',2*t/lambda];
        %KHMatrix = [K.*H ,1-p+eta_p-beta_p;(1-p+eta_p-beta_p)',2*t/lambda];
        cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p);
        cConstraint=[beta_p>=0,eta_p>=0,HMatrix>=0,KHMatrix>=0,diag(H)==yapp.*(yall-(K*w_o)),-p<=K*w_o<=p,p<=1,ONESn'*p<=n_o];
        cConstraint=[cConstraint,-yab <= yall-(K*w_o)<=yab];
        cConstraint=[cConstraint, yab>=1-p,yab<=1+p];
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            info.type=type;
            info.primal=true;
            info.primalW_ohr =value(w_o);
            info.Hprimal     = value(H);
            info.primalp     = value(p);
            info.primalw_oxT_i=(info.primalW_ohr)'*K;
            info.yPredicted  = sign(value(yall));
            info.trError     = sum(abs(info.yPredicted~=yapp))/n;
            dalpha = dual(cConstraint(1));
            ualpha = dual(cConstraint(2));
            if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
                display('error in computing primal variable alpha in twofuncsvm case 7');
                pause;
                return;
            end
            %info.w_oloss     = 1-yapp.*(K*info.primalW_ohr);
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
            info.primalWalpha=value(ualpha); 
            info.pObjective  = value(cObjective);
            info.primalwxT_i =(info.yPredicted.*info.primalWalpha)'*K;
        end    
    case 10 % convex relaxation for semisupervised (1-y_i w_o^T x_i) ( 1- y_i w^T x_i)_+
        H     =sdpvar(n,n);
        p     =sdpvar(n,1);
        w_o   =sdpvar(n,1);
        g     =sdpvar(n,1);
        %yun   =sdpvar(n-n_l,1);%unlabeled data labels
        n_l   =size(initL,2);
        n_u   =n-n_l;
        X     =sdpvar(n_u,2);
        
        beta_p=sdpvar(n,1);
        eta_p =sdpvar(n,1);
        t     =sdpvar(1,1);
        yun   =sdpvar(n_u,1);
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        yall  =sdpvar(n,1);
        yall(initL)     =yapp(initL);
        yall(setunlab)  =yun;

                
        HMatrix      = sdpvar(n+1,n+1);
        KHMatrix     = sdpvar(n+1,n+1);
        HMatrix      = [H,yall-(K*w_o);(yall-(K*w_o))',1];
        %KHMatrix = [K.*H ,1-yapp.*(K*w_o)+eta_p-beta_p;(1-yapp.*(K*w_o)+eta_p-beta_p)',2*t/lambda];
        %KHMatrix = [K.*H ,1-p+eta_p-beta_p;(1-p+eta_p-beta_p)',2*t/lambda];
        gpr=sdpvar(n,1);
        gpr(initL)   = diag(yall(initL))*K(initL,:)*w_o;
        gpr(setunlab)= X(:,1);
        KHMatrix = [K.*H ,1-gpr+eta_p-beta_p;(1-gpr+eta_p-beta_p)',2*t/lambda];
        cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p);
        cConstraint=[beta_p>=0,eta_p>=0,HMatrix>=0,KHMatrix>=0,diag(H)==1-p,-p<=K*w_o<=p,p<=1,ONESn'*p<=n_o];
        for i=1:n_u
            Z{i}=[1,X(i,1),yun(i);X(i,1),X(i,2),K(i,:)*w_o;yun(i),K(i,:)*w_o,1];
            cConstraint = [cConstraint,Z{i}>=0];
        end
        
        cConstraint = [cConstraint,X(:,1)>= K(setunlab,:)*w_o-2*p(setunlab)];
        cConstraint = [cConstraint,X(:,1)>= yall(setunlab)+K(setunlab,:)*w_o-1];
        cConstraint = [cConstraint,X(:,1)<= K(setunlab,:)*w_o+1-yall(setunlab)];
        cConstraint = [cConstraint,X(:,1)<= K(setunlab,:)*w_o+2*p(setunlab)];
        cConstraint = [cConstraint,X(:,1)>= yall(setunlab)+1-K(setunlab,:)*w_o];
        cConstraint = [cConstraint,X(:,1)>= 2*p(setunlab)-K(setunlab,:)*w_o];
        cConstraint = [cConstraint,X(:,1)>= -K(setunlab,:)*w_o-2*p(setunlab)];
        cConstraint = [cConstraint,X(:,1)>= -K(setunlab,:)*w_o-1-yall(setunlab)];
    
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            info.type=type;
            info.primal=true;
            info.primalW_ohr =value(w_o);
            info.Hprimal     = value(H);
            info.primalp     = value(p);
            info.primalw_oxT_i=(info.primalW_ohr)'*K;
            info.yPredicted  = sign(value(yall));
            info.trError     = sum(abs(info.yPredicted~=yapp))/n;
            dalpha = dual(cConstraint(1));
            ualpha = dual(cConstraint(2));
            if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
                display('error in computing primal variable alpha in twofuncsvm case 7');
                pause;
                return;
            end
            %info.w_oloss     = 1-yapp.*(K*info.primalW_ohr);
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
            info.primalWalpha=value(ualpha); 
            info.pObjective  = value(cObjective);
            info.primalwxT_i =(info.yPredicted.*info.primalWalpha)'*K;
        end
    case 11 % convex relaxation for supervised (1-y_i w_o^T x_i) ( 1- y_i w^T x_i)_+
        
        G     =sdpvar(n,n);
        p     =sdpvar(n,1);
        w_o_gy=sdpvar(n,1);
        w_o_g =sdpvar(n,1);
        w_o   =sdpvar(n,1);
        
%       w_o   =yapp.*w_o_gy+w_o_g;
        beta_p=sdpvar(n,1);
        eta_p =sdpvar(n,1);
        t     =sdpvar(1,1);
        YM    = diag(yapp);
        
        GMatrix  = sdpvar(n,n);
        KGMatrix = sdpvar(n,n);
        GMatrix  = [G,1-yapp.*(K*w_o);(1-yapp.*(K*w_o))',1];
        KGMatrix = [YM*K.*G*YM ,1-yapp.*(K*w_o)+eta_p-beta_p;(1-yapp.*(K*w_o)+eta_p-beta_p)',2*t/lambda];
        
        cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p);
        cConstraint=[beta_p>=0,eta_p>=0,GMatrix>=0,KGMatrix>=0,diag(G)==1-yapp.*(K*w_o),-p<=K*w_o<=p,p<=1,norm(p)^2 <=n_o];
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            info.type        =type;
            info.primal      =true;
            info.primalW_ohr =value(w_o);
            info.Gprimal     = value(G);
            info.primalp     = value(p);
            info.primalw_oxT_i=(info.primalW_ohr)'*K;  
            dalpha = dual(cConstraint(1));
            ualpha = dual(cConstraint(2));
            if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
                display('error in computing primal variable alpha in twofuncsvm case 7');
                pause;
                return;
            end
            info.w_oloss     = 1-yapp.*(K*info.primalW_ohr);
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
            info.primalWalpha=value(ualpha); 
            info.pObjective  = value(cObjective);
            info.primalwxT_i =(YM*info.primalWalpha.*info.w_oloss)'*K;
        end   
    case 12 % approximating (1-q_i)(1-y_iw_o^T\phi(x_i) by  min{1-q_i,1-y_i w_o^T\phi(x_i)} and convex relaxation and Alternating minimization for  YY^T
        n_l   =size(initL,2);
        n_u   =n-n_l;
        n_q    =n_u; % set Q to be all of unlabeled data 
        m=ones(n,1);
        q=zeros(n_q,1);
        G=ones(n+n_q,n+n_q);
        converged=false;
        [info]=LearnCRY(n,n_l,n_u,lambda,lambda_o,initL,yapp,K,G,m,q);
        ypre(1:n)=info.ysemd;
        ypre(n+1:n+n_q)=ones(n_q,1);
        betapre(1:n)=info.beta_p;%zeros(n+n_q,1);
        betapre(n+1,n+n_q)=zeros(n_q,1);
        etapre (1:n)=info.eta_p;  %zeros(n+n_q,1);
        etapre(n+1,n+n_q)=zeros(n_q,1);
        i=1;
        while ~converged 
            %call to learn f
            [info1]=LearnCRlrcoef(n,n_l,n_u,lambda,lambda_o,initL,ypre,K);% Learn all parameters except Y using above told approxiamtion
            betanew=info1.beta_p;
            etanew =info1.eta_p;
            %G=info1.G;
            G=info1.G;
            m=info1.m;
            q=info1.q;
            i=i+1
            del=norm(etanew-etapre)+norm(betanew-betapre)
            if del < 0.0001
               converged=true; 
            end
            betapre=betanew;
            etapre =etanew;
            %call to learn YY^T
            if ~converged 
                [info]=LearnCRY(n,n_l,n_u,lambda,lambda_o,initL,yapp,K,G,m,q);
                ypre=info.ysemd;
            end
        end
    case 13 % convex relaxation for semisupervised (1-y_i w_o^T x_i) ( 1- y_i w^T x_i)_+
            % based on max(Y-g',g'-Y) >= q-epsilon
        n_l=size(initL,2);
        n_u=n-n_l;
        n_q=n_u;
        nap=n+n_q;
        batchsize=2;
        G     =sdpvar(nap,nap);
        beta_p=sdpvar(nap,1);
        beta_pr=sdpvar(nap,1);
        eta_p =sdpvar(nap,1);
        eta_pr=sdpvar(nap,1);
        g=sdpvar(nap,1);
        gpr=sdpvar(nap,1);
        q=sdpvar(n_q,1);
        lo=sdpvar(n_q,1);
        mg=sdpvar(n_q,1);
        
        t     =sdpvar(1,1);
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        setQuery = setunlab;
        
        GMatrix  = sdpvar(nap+1,nap+1);
        KGMatrix = sdpvar(nap+1,nap+1);
        KQ   =[K,K(setall,setQuery);K(setall,setQuery)',K(setQuery,setQuery)];
        KGMatrix = [KQ.*G ,gpr+eta_pr-beta_pr;(gpr+eta_pr-beta_pr)',2*t/lambda];
        v=sdpvar(nap,1);
        yun=sdpvar(n_u,1);
        GMatrix  = [G,v;v',1];
        myeps = 0.3;
        cObjective = t+sum(beta_p)+sum(eta_p(n+1:nap));
        cConstraint=[beta_p>=0,eta_p>=0,GMatrix>=0,KGMatrix>=0,diag(G)==v,0<=q<=1,sum(q)==batchsize];
        cConstraint=[cConstraint,g(initL)==1;g(setunlab)==1-q;g(n+1:nap)==0;...
                        gpr(initL)==yapp(initL);gpr(n+1:nap)==0];
        cConstraint=[cConstraint,-q<=gpr(setunlab)-yun<=q;-1+q<=gpr(setunlab)<=1-q;...
                      lo>=q-myeps;lo>=0;mg>=gpr(setunlab)-yun;mg>=yun-gpr(setunlab);mg>=lo];
        cConstraint=[cConstraint,v(initL)==ones(n_l,1);v(setunlab)==ones(n_u,1)-q;v(n+1:nap) ==q];      
        cConstraint=[cConstraint,eta_pr(initL)==yapp(initL).*eta_p(initL);beta_pr(initL)==yapp(initL).*beta_p(initL)];
        cConstraint=[cConstraint,eta_pr(n+1:nap)==eta_p(n+1:nap);beta_pr(n+1:nap)==beta_p(n+1:nap)];
        cConstraint=[cConstraint,eta_pr(setunlab)-2*gpr(setunlab)<=2*(1-q)-eta_p(setunlab);2*gpr(setunlab)-eta_pr(setunlab)<=2*(1-q)-eta_p(setunlab)];
        cConstraint=[cConstraint,-2*(1-q)<=eta_pr(setunlab)<=2*(1-q);-2*(1-q)<=eta_p(setunlab)<=2*(1-q)];
        cConstraint=[cConstraint,beta_pr(setunlab)-2*gpr(setunlab)<=2*(1-q)-beta_p(setunlab);2*gpr(setunlab)-beta_pr(setunlab)<=2*(1-q)-beta_p(setunlab)];
        cConstraint=[cConstraint,-2*(1-q)<=beta_pr(setunlab)<=2*(1-q);-2*(1-q)<=beta_pr(setunlab)<=2*(1-q)];
        cConstraint=[cConstraint,-v<=g+eta_p-beta_p<=v];
        
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            info.beta_p=value(beta_p);
            info.beta_pr=value(beta_pr);
            info.eta_p=value(eta_p);
            info.eta_pr=value(eta_pr);
            info.g=value(g);
            info.gpr=value(gpr);
            info.q =value(q);
            info.objective=value(cObjective);
            info.GMatrix = value(GMatrix);
            info.v       =value(v);
            info.yun     =value(yun);
        end
    case 14 % Active Learning:min_V max_alpha
        n_l=size(initL,2);
        n_u=n-n_l;
        n_q=n_u;
        nap=n+n_q;
        bSize=1;
        V     =sdpvar(nap+1,nap+1);
        z_u=sdpvar(n_u,1);
        q=sdpvar(n_q,1);
        alpha_til   =sdpvar(nap,1);
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        setQuery = setunlab;
        % move elements in K so 1:n_l labeled data,n_l+1:n_l+n_u:unlabeled
        % data, and n_l+n_u+1:n_l+n_u+n_q: query copies of data.
        y_l=yapp(initL);
        K_m(1:n_l,1:n_l)   = K(initL,initL);
        K_m(1:n_l,n_l+1:n_l+n_u) = K(initL,setunlab);
        K_m(n_l+1:n_l+n_u,1:n_l) = K(setunlab,initL); 
        K_m(n_l+1:n_l+n_u,n_l+1:n_l+n_u)=K(setunlab,setunlab);
        K_m(1:n,n+1:nap  ) = K(setall,setQuery);
        K_m(n+1:nap,1:n)   = K_m(1:n,n+1:nap  )';
        K_m(n+1:nap,n+1:nap)= K(setQuery,setQuery);
        
        %starting point : start from a nonfeasible point
        V_primal= ones(nap,1)*ones(nap,1)';
        V_pre   = V_primal;
        qprimal=ones(n_q,1);
        alpha_pre = zeros(nap,1);
        alpha_tval=rand(nap,1);
        converged = false;
        while ~converged
            %solve outer minimization problem
            R=K_m.*(alpha_tval*alpha_tval');
            V_primal = value(V(1:nap,1:nap));
            V_in=sdpvar(nap,nap);
            z_q =sdpvar(n_q,1);
            z_u =sdpvar(n_u,1);
            zall=sdpvar(nap,1);
            zall=[y_l;z_u;z_q];
            V = [ V_in,zall;zall',1];
            [B,T]=eig(R);
            R=B*diag(max(diag(T),0))*B';
            %constraints
%             cConstraint = [V>=0;V(1:n_l,nap+1)==y_l;0<=V(nap+1,n_l+n_u+1:nap)<=1,V(nap+1,nap+1)==1];    
%             cConstraint = [cConstraint;sum(V(nap+1,n+1:nap))==bSize];%;sum(V(1,n+1:nap))==bSize
%             cConstraint = [cConstraint;diag(V(1:n_l,1:n_l))==1];
            cConstraint = [V>=0;diag(V)<=1;z_q>=0;z_q<=1;sum(z_q)==1;diag(V(1:n_l,1:n_l))==1];%
            cConstraint = [cConstraint,diag(V(n_l+1:n,n_l+1:n))==diag(V(n+1:nap,n+1:nap))-2*z_q+ones(n_q,1)]; 
            cConstraint = [cConstraint,diag(V(n_l+1:n,n_l+1:n))+diag(V(n+1:nap,n+1:nap))==ones(n_q,1)];
%             for i=1:n_u
%                 cConstraint = [cConstraint,V(n_l+i,n_l+i)+V(n+i,n+i)==1;...
%                                V(n_l+i,n_l+i)== V(n+i,n+i)-2*V(nap+1,n+i)+1];
%             end
            outerObjective = -sum(alpha_tval(n_l+1:n).*V(n+1:nap,nap+1))...
                             -1/(2*lambda)*trace(V(1:nap,1:nap)*R);
            sol = optimize(cConstraint,outerObjective);
            if sol.problem==0
                V_primal = value(V(1:nap,1:nap));
                VT=value(V);
                qprimal = value(V(n+1:nap,nap+1));
                objval=value(outerObjective);
            end  
            [U,D]=eig(V_primal);
            V_primalpr=U*diag(max(diag(D),0))*U';
            % solve inner maximization problem
            %define objective and constraint to YALMIP in minimization form
            innerObjective = -sum(alpha_til(1:n))+sum(alpha_til(n_l+1:n).*qprimal)...
                             +1/(2*lambda)*(alpha_til'*V_primalpr*alpha_til);
            cConstraint=[alpha_til(1:n)>=0,alpha_til<=1,alpha_til(n+1:nap)>=-1];
            % solve inner maximization problem
            sol = optimize(cConstraint,innerObjective);
            if sol.problem==0
                alpha_tval=value(alpha_til);
                objval=-value(innerObjective);
            end
          
            dalpha=norm(alpha_tval-alpha_pre);
            dV    =norm(V_primal-V_pre);
            if dV< 0.001
                converged=true;
            end
            alpha_pre=alpha_tval;
            V_pre    =V_primal;
        end
    case 15 % Active Learning Relaxation without outlier detection. 
        c=1;%parameter for finding absolute value of y_u.*(1-q)
        n_l=size(initL,2);
        n_u=n-n_l;
        n_q=n_u;
        nap=n+n_q;
        bSize=1;
        V     =sdpvar(nap+1,nap+1);
        q=sdpvar(n_u,1);
        r=sdpvar(n_u,1);
        beta_p=sdpvar(nap,1);
        eta_p =sdpvar(nap,1);
        t     =sdpvar(1,1);
        
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        setQuery =n+1:nap;
        
        y_l     =yapp(initL);
                
        KVMatrix     = sdpvar(nap+1,nap+1);
        s=sdpvar(nap,1);
        
        KQ   =[K,K(setall,setunlab);K(setall,setunlab)',K(setunlab,setunlab)];
        KVMatrix = [KQ.*V(1:nap,1:nap) ,1-s+eta_p-beta_p;(1-s+eta_p-beta_p)',2*t/lambda];
        cObjective = t+sum(beta_p)+sum(eta_p(n+1:nap))+c*sum(r);
        
        cConstraint=[beta_p>=0,eta_p>=0,V>=0,KVMatrix>=0];
        cConstraint=[cConstraint,sum(q)==bSize,0<=q,q<=1];
        cConstraint=[cConstraint,V(nap+1,nap+1)==1,diag(V(1:nap,1:nap))<=1,V(initL,nap+1)==y_l];
        cConstraint=[cConstraint,r>=V(n_l+1:n,nap+1),r>=-V(n_l+1:n,nap+1),r+q==1];
        cConstraint=[cConstraint,V(n+1:nap,nap+1)==q,...
                     s(initL)==zeros(n_l,1),s(n+1:nap)==zeros(n_q,1),s(setunlab)==q];
        
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))+diag(V(setQuery,setQuery))==ones(n_q,1)];
%         cConstraint=[cConstraint,diag(V(setunlab,setunlab))==diag(V(setQuery,setQuery))-2*q(setunlab)+ones(n_q,1)];
        
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            info.type=type;
            info.primal=true;
            info.Vprimal     = value(V);
            info.primalq     = value(q);
            info.qyu   =value(V(n_l+1:n,nap+1));% y_u_i * (1-q_i)
            info.r    = value(r);%absoulte value of qyu
            info.s    =value(s);
            [maxq,imaxq]=max(info.primalq);
            info.qbatch=imaxq;
            info.yPredicted  = sign(value(info.qyu));
            info.trError     = sum(abs(info.yPredicted~=yapp(setunlab)))/n;
            dalpha = dual(cConstraint(1));% dual variable for beta_p
            ualpha = dual(cConstraint(2));% dual variable for eta_p
            if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
                display('error in computing primal variable alpha in twofuncsvm case 7');
                pause;
                return;
            end
            %info.w_oloss     = 1-yapp.*(K*info.primalW_ohr);
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
            info.primalWalpha=value(ualpha); 
            info.pObjective  = value(cObjective);
            info.primalwxT_i =(info.yPredicted.*info.primalWalpha)'*K;
        end
    case 16 % case 1 except with another Kernel function for w_o
        outlierind=n;
        valo=zeros(n,1);
        valo(outlierind)=1;
        alpha_r=sdpvar(n,1);
        h=sdpvar(n,1);
        r=sdpvar(n,1);
        g=sdpvar(n,1);
        t=sdpvar(1,1);
        s=sdpvar(n,1);

        s          =alpha_r.*yapp+h-r;
        fnormterm  =sdpvar(1,1);
        w_onormterm=sdpvar(1,1);

        fnormterm  = alpha_r'*Ky*alpha_r;
        w_onormterm= s'*K_o*s;

        Objective= fnormterm/(2*lambda)+w_onormterm/(2*lambda_o)-ONESn'*alpha_r+ONESn'*g+n_o*t;
        
        Constraints = [alpha_r>=0,alpha_r<= 1,h>=0,r>=0,g>=0,h+r==g+t*ONESn];%contraint about t is removed. 

        sol=optimize(Constraints,Objective);
        
        info.solproblem = sol.problem;
        if sol.problem==0 
             info.type  =type;
            info.alpha_r=value(alpha_r);
            info.h = value(h);
            info.r = value(r);
            info.t = value(t);
            info.g = value(g);
            info.objval=value(Objective);
            alphnorm=value(fnormterm);
            hrknorm =value(w_onormterm);
            info.normW = alphnorm/(lambda^2);
            info.normW_o = hrknorm/(lambda_o^2);
            info.SigAlpha= ONESn'*info.alpha_r;
            info.w_oTx_i=(info.alpha_r.*yapp+info.h-info.r)'*K_o/lambda_o;
            info.wTx_i  =(info.alpha_r.*yapp)'*K/lambda;
            sumpi       =sum(abs(info.w_oTx_i));
            info.xi     =max(ONESn-yapp.*(info.w_oTx_i+info.wTx_i)',0);
            xisum       =sum(info.xi);
            pstar=lambda*info.normW/2+lambda_o*info.normW_o/2+xisum;
            dstar=sum(info.alpha_r)-lambda*info.normW/2-lambda_o*info.normW_o/2-sum(info.g)-n_o*info.t;
            strabs = sprintf('Sum of |w_o^T*x_i:%5.2f vs:%5.2f',sumpi,n_o);
            display(strabs);
        else
            display('Problem is not solved successfully');
        end
        %Solving primal problem directly using YALMIP
        a=1;
        walpha_p =sdpvar(n,1);
        w_ohr_p  =sdpvar(n,1);
        xi       =sdpvar(n,1);
        p        =sdpvar(n,1);
%        gwof     =sdpvar(n,1);
        up       =1;
        down     =-1;
        pObjective =sum(xi)+lambda*walpha_p'*K*walpha_p/2+lambda_o*w_ohr_p'*K_o*w_ohr_p/2;
        pConstraint=[walpha_p>=down,walpha_p<=up,xi>=0,xi>=1-yapp.*(K*walpha_p+K*w_ohr_p),...
                     -p<=K*w_ohr_p<=p,p<=1,ONESn'*p==n_o]%,K(14,:)*w_ohr_p==0.8,K(15,:)*w_ohr_p==-0.8]%,...
%                      gwof>=K*walpha_p+a*K*w_ohr_p-a,gwof<=a+K*walpha_p-a*K*w_ohr_p,...
%                      gwof<=a-K*walpha_p+a*K*w_ohr_p,gwof>=-a-K*walpha_p-a*K*w_ohr_p,...
%                      gwof'*ONESn>=-a*n_o,gwof'*ONESn<=a*n_o];%,gwof'*ONESn>=-ONESn'*K*walpha_p,gwof'*ONESn<=ONESn'*K*walpha_p
        sol=optimize(pConstraint,pObjective);
        if sol.problem==0
            info.primalWalpha=value(walpha_p);
            info.pObjective  = value(pObjective);
            info.primalW_ohr =value(w_ohr_p);
            info.primalxi    = value(xi);
            info.primalp     = value(p);
            info.primalwxT_i =info.primalWalpha'*K;
            info.primalw_oxT_i=info.primalW_ohr'*K;
            info.outliertest.f_o  =info.primalw_oxT_i(14);
            info.outliertest.f    =info.primalwxT_i(14);   
%             info.gwof        =value(gwof);
        else 
            display('cannot solve the problem');
        end
        info.primal=true;  
    
    otherwise
        display('wrong type');
end

end
function [info]=LearnCRlrcoef(n,n_l,n_u,lambda,lambda_o,initL,ypre,K)
        M=1;
        n_q=n_u;
        n_o=2;%assume there is only n_o outlier in the data
        LG     =sdpvar(n+n_q+1,n+n_q+1);
        KG     =sdpvar(n+n_q+1,n+n_q+1);
        p     =sdpvar(n,1);
        w_o   =sdpvar(n,1);
        q     =sdpvar(n_q,1); %variable for active learning selection
        m     = sdpvar(n,1); % variable equal min{1-q_i,1-y_i w_o^T\phi(x_i)} to approximate (1-q_i)(1-y_iw_o^T\phi(x_i)
        G     =sdpvar(n+n_q,n+n_q);
        beta_p=sdpvar(n+n_q,1);
        eta_p =sdpvar(n+n_q,1);
        t     =sdpvar(1,1);
        
        setall   =1:n;
        setunlab =setdiff(setall,initL);
        setQuery = setunlab;
        ap   =[m;q];  
        mtil =[m;zeros(n_q,1)];
        ml   =sdpvar(n,1);
        ml   =1-K*w_o;% for labeled data m=1-K*w_o, because 1-q_i=1
        YM=ypre*ypre';
        KQ   =[K,K(setall,setQuery);K(setall,setQuery)',K(setQuery,setQuery)];
        LG      = [G,ap;ap',1]; %matrix with [G,(m,q);(m,q),1]
        KG = [KQ.*G.*YM ,mtil+eta_p-beta_p;(mtil+eta_p-beta_p)',2*t/lambda];
        save('yalmip.mat','K','ypre','initL','lambda','lambda_o','n','n_u','n_q');
        cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p)+sum(eta_p(n+1:n+n_q))-M*sum(m);
        cConstraint=[beta_p>=0,eta_p>=0,LG>=0,KG>=0,G>=0,diag(G)==[1-K*w_o;ones(n_q,1)],-p<=K*w_o<=p,p<=1,sum(p)==n_o];
        cConstraint=[cConstraint,m(initL)==ml(initL),m(setQuery)<=1-q,m(setQuery)<=ml(setQuery),0<=q<=1,p(setQuery)+q<=1,sum(q)==1];
        sol = optimize(cConstraint,cObjective);
        if sol.problem==0
            info.primal=true;
            info.primalW_ohr = value(w_o);
            info.G           = value(G);
            info.primalw_oxT_i=(info.primalW_ohr)'*K;
            dalpha = dual(cConstraint(1));
            ualpha = dual(cConstraint(2));
%             if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
%                 display('error in computing primal variable alpha in twofuncsvm case 7');
%                 pause;
%                 return;
%             end
            info.w_oloss     = 1-ypre(setall).*(K*info.primalW_ohr);
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM

            info.pObjective  = value(cObjective);
            %info.primalwxT_i =(ypre.*info.primalWalpha)'*K;
            info.m    = value(m);
            info.beta_p=value(beta_p);
            info.eta_p=value(eta_p);
            info.primalp     = value(p);
            info.q           =value(q);
        end
end
function [info]=LearnCRY(n,n_l,n_u,lambda,lambda_o,initL,yapp,K,G,m,q)
        %n_q = n_u;
        
        setall   =1:n;
        yun      =sdpvar(n_u,1);
        setunlab =setdiff(setall,initL);
        %setQuery =setdiff(setall,initL);%select query from all of unlabeled data
        yall  =sdpvar(n,1);
        yall(initL)     =yapp(initL);
        yall(setunlab)  =yun;
        %yall(n+1:n+n_q) = ones(n_q,1);
        yab             =sdpvar(n,1);
        
        %LG     =sdpvar(n+n_q+1,n+n_q+1);
        KG     =sdpvar(n+1,n+1);
        %p     =sdpvar(n,1);
        %w_o   =sdpvar(n,1);
        %q     =sdpvar(n,1); %variable for active learning selection
        %m = sdpvar(n,1); % variable equal min{1-q_i,1-y_i w_o^T\phi(x_i)} to approximate (1-q_i)(1-y_iw_o^T\phi(x_i)
        %G    =sdpvar(n+n_q,n+n_q);
        beta_p=sdpvar(n,1);
        eta_p =sdpvar(n,1);
        t     =sdpvar(1,1);
        
        %mq   =[m;q];  
        %mtil =[m;zeros(n_q,1)];
        %KQ   =[K,K(setall,setQuery);K(setall,setQuery)',K(setQuery,setQuery)];
        %YM=ypre*ypre';
        YM=sdpvar(n,n);
        GD=G(setall,setall);
        %LG      = [G,mq;mq',1]; %matrix with [G,(m,q);(m,q),1]
        KG = [K.*GD.*YM ,m+eta_p-beta_p;(m+eta_p-beta_p)',2*t/lambda];
        YL = [YM,yall;yall',1];
        cObjective = t+sum(beta_p)+sum(eta_p(n+1:n+n_q));
        cConstraint=[beta_p>=0,eta_p>=0,KG>=0,YL>=0,diag(YM)==1];
        sol = optimize(cConstraint,cObjective);
        
        if sol.problem==0
            info.primal=true;
            dalpha = dual(cConstraint(1));
            ualpha = dual(cConstraint(2));
%             if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
%                 display('error in computing primal variable alpha in twofuncsvm case 7');
%                 pause;
%                 return;
%             end
           
            %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
           
            info.pObjective  = value(cObjective);
            info.beta_p=value(beta_p);
            info.eta_p=value(eta_p);
            info.ysemf =value(yall);
            info.ysemd =sign(info.ysemf);
        end
end
%         M=1;
%         n_q=n_u;
%         n_o=2;%assume there is only n_o outlier in the data
%         LG     =sdpvar(n+n_q+1,n+n_q+1);
%         KG     =sdpvar(n+n_q+1,n+n_q+1);
%         p     =sdpvar(n,1);
%         w_o   =sdpvar(n,1);
%         q     =sdpvar(n_q,1); %variable for active learning selection
%         m     = sdpvar(n,1); % variable equal min{1-q_i,1-y_i w_o^T\phi(x_i)} to approximate (1-q_i)(1-y_iw_o^T\phi(x_i)
%         G     =sdpvar(n+n_q,n+n_q);
%         beta_p=sdpvar(n+n_q,1);
%         eta_p =sdpvar(n+n_q,1);
%         t     =sdpvar(1,1);
%         
%         setall   =1:n;
%         setunlab =setdiff(setall,initL);
%         setQuery = setunlab;
%         ap   =[m;q];  
%         mtil =[m;zeros(n_q,1)];
%         ml   =sdpvar(n,1);
%         ml   =1-K*w_o;% for labeled data m=1-K*w_o, because 1-q_i=1
%         YM=ypre*ypre';
%         KQ   =[K,K(setall,setQuery);K(setall,setQuery)',K(setQuery,setQuery)];
%         LG      = [G,ap;ap',1]; %matrix with [G,(m,q);(m,q),1]
%         KG = [KQ.*G.*YM ,mtil+eta_p-beta_p;(mtil+eta_p-beta_p)',2*t/lambda];
%         save('yalmip.mat','K','ypre','initL','lambda','lambda_o','n','n_u','n_q');
%         cObjective = t+lambda_o*w_o'*K*w_o/2+sum(beta_p)+sum(eta_p(n+1:n+n_q))-M*sum(m);
%         cConstraint=[beta_p>=0,eta_p>=0,LG>=0,KG>=0,G>=0,diag(G)==[1-K*w_o;ones(n_q,1)],-p<=K*w_o<=p,p<=1,sum(p)==n_o];
%         cConstraint=[cConstraint,m(initL)==ml(initL),m(setQuery)<=1-q,m(setQuery)<=ml(setQuery),0<=q<=1,p(setQuery)+q<=1,sum(q)==1];
%         sol = optimize(cConstraint,cObjective);
%         if sol.problem==0
%             info.primal=true;
%             info.primalW_ohr = value(w_o);
%             info.G           = value(G);
%             info.primalw_oxT_i=(info.primalW_ohr)'*K;
%             dalpha = dual(cConstraint(1));
%             ualpha = dual(cConstraint(2));
% %             if sum(abs(abs(1-dalpha)-ualpha)) >0.0001
% %                 display('error in computing primal variable alpha in twofuncsvm case 7');
% %                 pause;
% %                 return;
% %             end
%             info.w_oloss     = 1-ypre(setall).*(K*info.primalW_ohr);
%             %l(y_i w_o^T\phi(x_i)) must be multiplied to alpha to obtain alpha for SVM
% 
%             info.pObjective  = value(cObjective);
%             %info.primalwxT_i =(ypre.*info.primalWalpha)'*K;
%             info.m    = value(m);
%             info.beta_p=value(beta_p);
%             info.eta_p=value(eta_p);
%             info.primalp     = value(p);
%             info.q           =value(q);
%         end
