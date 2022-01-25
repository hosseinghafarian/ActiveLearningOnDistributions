function [equalitydiff, eq, Inequality, Ineq]      = ConstraintsCheck(x_k, learningparams)
%% Checks whether or not x_k is satisfying Constraints
%  returns : 
%            equalitydiff: sum of norm of difference in equality
%            eq          : details by each equality constraint
%            Inequality  : is true if all of inequality constraints are satisfyed
%            Ineq        : details by each inequality constraint
global cnstData;
      nSDP      = cnstData.nSDP;
      n_u       = cnstData.n_u;
      n_q       = cnstData.n_q;
      n_l       = cnstData.n_l;
      out_lab_sep = cnstData.label_outlier_seperate_deal;
      if  out_lab_sep == false
         n_o       = cnstData.n_o;
      else
         lnoiseper = cnstData.lnoiseper;
         onoiseper = cnstData.onoiseper;
      end
      batchSize = cnstData.batchSize;
      unlabeled = cnstData.unlabeled;
      
      initL     = cnstData.initL(cnstData.initL>0);
      Yl        = cnstData.Yl(initL);
      Kernel    = cnstData.K;
      setunlab  = cnstData.unlabeled';
      extendInd = cnstData.extendInd;
      nap       = cnstData.nap;
      
      nLM       = nSDP*nSDP;
      
      [G_plus,p,q,qyu,w_obeta,st]  = getParts(x_k);
      rl        = G_plus(initL,nSDP);
      r         = ones(n_u,1)-p(unlabeled)-q; % attention: we removed this variable so checking it is trivail
      g_D(initL)              = p(initL);          % attention: we removed this variable so checking it is trivail
      g_D(cnstData.extendInd) = zeros(n_q,1);      % attention: we removed this variable so checking it is trivail
      g_D(setunlab)           = 1-r;               % attention: we removed this variable so checking it is trivail
      
     %cConstraint1 = [beta_p>=0,eta_p>=0,G_plus>=0,KVMatrix>=0]; %nonnegativity constraints
                                        %ok
      eq.cSumq       = norm(sum(q)-batchSize);
      eq.cNapNap     = norm(G_plus(nap+1,nap+1)-1);
      eq.cdiag_unlab = norm(diag(G_plus(setunlab,setunlab))-r);
      eq.cdiag_initL = norm(diag(G_plus(initL,initL))-1+p(initL));
      eq.cdiag_query = norm(diag(G_plus(extendInd,extendInd))-q);
      eq.cqExtendInd = norm(G_plus(extendInd,nap+1)-q);
      eq.crlKw_obeta = norm(rl-Yl+Kernel(initL,:)*w_obeta);
      eq.crqpEquality= norm(r+q+p(unlabeled)-1);
      equalitydiff   = eq.cSumq + eq.cNapNap + eq.cdiag_unlab + eq.cdiag_initL + eq.cdiag_query + eq.cqExtendInd+ ...
                       eq.crlKw_obeta + eq.crqpEquality;
%       
%       Ineq.cqInRange = [0<=q,q<=1];% constraints on q%%implict 
% %      Ineq.cqInRange = ~Ineq.cqInRange; 
%       %Ineq.cpKernel  = [-p<=Kernel*w_obeta;Kernel*w_obeta<=p;p<=1;];
% %      Ineq.cpKernel  = ~Ineq.cpKernel;               
%       if out_lab_sep
%           Ineq.cSump = ~(sum(p(initL))<=lnoiseper*n_l/100 && sum(p(setunlab)<=onoiseper*n_u/100));
%       else
%           Ineq.cSump     = ~(sum(p)<n_o);        
%       end
%       Ineq.crunlab   = [r>=G_plus(unlabeled,nap+1);
%                           r>=-G_plus(unlabeled,nap+1)];
% %      Ineq.crunlab   = ~Ineq.crunlab;
%       Inequality     = AndElements(Ineq.cqInRange)&AndElements(Ineq.cSump)&...  %AndElements(Ineq.cpKernel)&
%                        AndElements(Ineq.crunlab(1));
      Ineq.cqInRange = norm(max(-q,0))+norm(max(q-1,0));% constraints on q%%implict 
%      Ineq.cqInRange = ~Ineq.cqInRange; 
      %Ineq.cpKernel  = [-p<=Kernel*w_obeta;Kernel*w_obeta<=p;p<=1;];
%      Ineq.cpKernel  = ~Ineq.cpKernel;               
      if out_lab_sep
          Ineq.cSump = max(sum(p(initL))-lnoiseper*n_l/100,0)+max(sum(p(setunlab)-onoiseper*n_u/100),0);
      else
          Ineq.cSump     = max(sum(p)-n_o,0);        
      end
      Ineq.crunlab   = norm(max(-r+G_plus(unlabeled,nap+1),0))+norm(max(-r-G_plus(unlabeled,nap+1),0));
%      Ineq.crunlab   = ~Ineq.crunlab;
      Inequality     = Ineq.cqInRange + Ineq.cSump + Ineq.crunlab;             
end
function [logicres] = AndElements(mat)
    [n,m]     = size(mat);
    logicres  = true;
    for i = 1: n
        for j = 1:m
            logicres = logicres&mat(i,j);
        end
    end
end