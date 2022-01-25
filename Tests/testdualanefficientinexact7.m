function docode
%% this is the complete form of the test for primal dual problem, with both w_o and all other variables.
% It works fine. The primal and dual objectives are completely equal (
% 10-5) and the corresponding constraints in the dual are satisfied. 
close all
recomp=0;
%% Loading data and computing constraints matrices
load('data4test','lambda','K','KE','alphapre','alphatild','nSDP','n_S','nSDP','n_l','n_u','initL','unlabeled','query','Yl','batchSize','n_o','n_lbn');
lambda_o = lambda/10;

[A_EC,b_EC,A_IC,s_IC,A_EV,b_EV,A_IV,s_IV,B_EV,B_IV,y_EC,y_IC,y_EV,y_IV,E_E,E_I] = getConstraints2(K,nSDP,n_l,n_u,n_S,initL,unlabeled,query,Yl,batchSize,n_o,n_lbn);

A_EC = A_EC(1:nSDP*nSDP+n_S,:);
A_IC = A_IC(1:nSDP*nSDP+n_S,:);
A_EV = A_EV(1:nSDP*nSDP+n_S,:);
A_IV = A_IV(1:nSDP*nSDP+n_S,:);

alphanorm  = norm(alphapre);
n_u        = nSDP-n_S-1;
%for now
query = 19:34;

%% Computing Coefficeint: c_k
c_a   =2;
% in the following lines since g is equal to p+q, coefficeints in
% optimization is computed based on that
C_kMS         = [(alphapre*alphapre').*KE/(2*lambda),zeros(nSDP-1,1);zeros(1,nSDP-1),0];% coefficients for G
alphatild     = alphapre(1:n_S);
n_q       = numel(query);
q_ind     = [repmat(nSDP,n_q,1),query'];  % this is the indexes of q
% coefficients for g (which substituted by g=p+q) so, it is considered for q
% and p seperately. alphatild'*(1-g)=alphatild'*1-alphatild'*q-alphatild'*q 
C_kMq         = sparse([q_ind(:,1)',q_ind(:,2)'],[q_ind(:,2)',q_ind(:,1)'],[alphatild(unlabeled)/2',alphatild(unlabeled)/2']);
C_kM    = C_kMS + C_kMq;
cp   = alphatild+c_a*ones(n_S,1);
c_k  = [reshape(C_kM,nSDP*nSDP,1);cp];%ca;cg];%first part for C_kM, the other parts for pag: p,a,g

n_IC = size(s_IC,1);
n_IV = size(s_IV,1);%2*n_S;%2*n_S+
A_IC     = A_IC';
s_IC = s_IC(1:n_IC);
A_IV     = A_IV';
s_IV = s_IV(1:n_IV);
A_EC = A_EC';
A_EV = A_EV';
%% scaling data
normA_EC = norm(A_EC,'fro');
normA_EV = norm(A_EV,'fro');
normA_IC = norm(A_IC,'fro');
normA_IV = norm(A_IV,'fro');
normC_kM = norm(C_kM,'fro');
normcp   = norm(cp);


maxCscale= max(normC_kM,normcp);
scaleA_EC = 1;
scaleA_EV = 1;
scaleA_IC = 1;
scaleA_IV = 1;
scaleC_kM = 1;
scalecp   = 1;
scaleca   = 1;
scalecg   = 1;
scaleG = 1;%norm(Gr,'fro');
maxEscale = max(normA_EC,normA_EV);
maxIscale = max(normA_IC,normA_IV);

% if maxEscale > maxIscale %&& maxEscale> maxCscale 
%     scaleA_IC = maxIscale/maxEscale;
% else%if maxIscale > maxEscale&& maxIscale > maxCscale
%     scaleA_EC = maxEscale/maxIscale;
% %     scaleA_IV = maxEscale/maxIscale;
% end
n_EC = size(b_EC,1);
n_EV = size(b_EV,1);
%% Setting Proximal parameters 
%setting proximal x^k for now, just fot the test
s_t= zeros(n_IC+n_IV,1);
u_t  = [zeros(nSDP*nSDP,1);zeros(n_S,1)]; 
w_obetat = zeros(n_S,1);

arho = 1;
Q = eye(n_S);
%% Primal problem 
A_EC = A_EC/scaleA_EC;
b_EC = b_EC/scaleA_EC;
A_EV = A_EV/scaleA_EV;
b_EV = b_EV/scaleA_EV;
A_IC     = A_IC/scaleA_IC;
s_IC     = s_IC/scaleA_IC;
X = sdpvar(nSDP,nSDP);
p = sdpvar(n_S,1);
a = sdpvar(n_S,1);
g = sdpvar(n_S,1);
w_obeta = sdpvar(n_S,1);
Xr= [reshape(X,nSDP*nSDP,1);p;];%a;g];
s1 = sdpvar(n_IC,1);
s2 = sdpvar(n_IV,1);
s  = [s1;s2];
cConstraint = [X>=0,p>=0,A_EC*Xr==b_EC,A_IC*Xr==s1,A_IV*Xr==s2+B_IV*w_obeta,A_EV*Xr==b_EV+B_EV*w_obeta,s1<=s_IC,s2<=s_IV];%,X(nSDP,query)>=0,X(nSDP,query)>=0
cObjective  = -c_k'*Xr+arho/2*norm(Xr-u_t)^2+arho/2*norm(s-s_t)^2 ...
              +lambda_o/2*w_obeta'*K*w_obeta+arho/2*(w_obeta-w_obetat)'*Q*(w_obeta-w_obetat);
if recomp==1
    sol = optimize ( cConstraint,cObjective);
    if sol.problem == 0 
        pobj   = value(cObjective);
        Xp     = value(X)/scaleC_kM;
        sp     = value(s);
        sp2    = value(s2);
        pp     = value(p)/scalecp;
        fvpAEC = A_EC*value(Xr)-b_EC;
        fvpAEV = A_EV*value(Xr)-b_EV;
        Sdup   = dual(cConstraint(1));
        pdup   = dual(cConstraint(2));
        AECdup = -dual(cConstraint(3)); % negative because of the direction of the constraint
        AICdup = -dual(cConstraint(4));
        AIVdup = -dual(cConstraint(5));
        AEVdup = -dual(cConstraint(6));% negative because of the direction of the constraint
        save('Pvals','pobj','Xp','sp','pp','AECdup','AICdup','AIVdup','AEVdup','Sdup');%,'pa','pg');
    end
else
    load('Pvals','pobj','Xp','sp','pp','AECdup','AICdup','AIVdup','AEVdup','Sdup');
end
%% Dual problem 
KQinv = inv(lambda_o*K+arho*Q);
y_EC = sdpvar(n_EC,1);
y_EV = sdpvar(n_EV,1);
y_IC = sdpvar(n_IC,1);
y_IV = sdpvar(n_IV,1);
Stil = sdpvar(nSDP,nSDP);
pdu  = sdpvar(n_S,1);
Z    = sdpvar(nSDP,nSDP);
zp   = sdpvar(n_S,1);
za   = sdpvar(n_S,1);
zg   = sdpvar(n_S,1);
Zr   = [reshape(Z,nSDP*nSDP,1);zeros(n_S,1)]; %
v    = sdpvar(n_IC+n_IV,1);
y_I  = [y_IC;y_IV];
s_I  = [s_IC;s_IV];
S = [reshape(Stil,nSDP*nSDP,1);pdu];%;adu;gdu];
cConstraint= [ Stil>=0,pdu>=0,v>=0];%,v<=arho*(s_I-gprox)+y_I];%,Z(1:n_S,:)==0,Z(:,1:n_S)==0,Z(n_S+1:nSDP,n_S+1:nSDP)>=0];%,Z(nSDP,query)>=0];
cObjective = -(b_EC'*y_EC+b_EV'*y_EV) ...
             + arho/2*norm(A_EC'*y_EC+A_IC'*y_IC+A_EV'*y_EV+A_IV'*y_IV+S+c_k+u_t/arho)^2 ...
             +1/2*(B_EV'*y_EV+B_IV'*y_IV+arho*Q*w_obetat)'*KQinv*(B_EV'*y_EV+B_IV'*y_IV+arho*Q*w_obetat)-lambda_o/2*w_obetat'*K*w_obetat ...
             +arho/2*norm(v+y_I-s_t/arho)^2+v'*s_I-1/(2*arho)*norm(u_t,'fro')^2-1/(2*arho)*norm(s_t)^2;
               
sol = optimize(cConstraint,cObjective);
if sol.problem==0 
   dobj = value(cObjective);
   dobj+pobj
   Sd = value(Stil);
   vd = value(v);
   Zd     = value(Z);
   y_ICd  = value(y_IC);
   y_IVd  = value(y_IV);
   y_ECd  = value(y_EC); 
   y_EVd  = value(y_EV);
   A_yECD = value(A_EC'*y_ECd);
   A_yEVD = value(A_EV'*y_EVd);
   A_yICD = value(A_IC'*y_ICd);
   % Although y_ECd and y_EVd are so much differnt from the primal computed
   % counterpart, they are correct since the answer is feasible, it's
   % objective function is very near to the optimal (just 1*10^-4
   % difference) and all the constraints are satisfied. 
   norm(AECdup-y_ECd)/norm(y_ECd)
   norm(AICdup-y_ICd)/norm(y_ICd)
   norm(AIVdup-y_IVd)/norm(y_IVd)
   norm(AEVdup-y_EVd)/norm(y_EVd)
   Xgr    = u_t+1/arho*(c_k+value(S)+A_EC'*y_ECd/scaleA_EC+A_IC'*y_ICd/scaleA_IC ...
                           +A_EV'*y_EVd/scaleA_EV+A_IV'*y_IVd/scaleA_IV);
%    Xgr2    = c_k+u_t+value(S)+A_EC'*AECdup/scaleA_EC+A_IC'*y_ICd/scaleA_IC+A_EV'*AEVdup/scaleA_EV+A_IV'*y_IVd/scaleA_IV;
%    norm(Xgr-Xgr2)
   w_obetav = value(KQinv*(B_EV'*y_EV+B_IV'*y_IV+arho*Q*w_obetat));
   fvAEC  = A_EC*Xgr-b_EC;
   fvAEV  = A_EV*Xgr-b_EV+B_EV*w_obetav;
   fvAIC  = max(-s_IC+A_IC*Xgr,0);
   fvAIV  = max(-s_IV+A_IV*Xgr+B_IV*w_obetav,0);
   feasAEC= norm(A_EC*Xgr-b_EC)
   feasAIV= norm(max(-s_IV+A_IV*Xgr,0))
   feasAEV= norm(A_EV*Xgr-b_EV)
   feasAIC= norm(max(-s_IC+A_IC*Xgr,0))
   Xg     = reshape(Xgr(1:nSDP*nSDP,1),nSDP,nSDP);
   mdim   = nSDP*nSDP;
   pd     = Xgr(mdim+1:mdim+n_S);
   

    trace(Sdup*Xg)
    norm(value(pdu)-pdup)/norm(pdup)


   
   norm(u_t,'fro')
   norm(value(S))
   norm(A_EC'*y_ECd/scaleA_EC)
   norm(A_IC'*y_ICd/scaleA_IC)
   norm(A_EV'*y_EVd/scaleA_EV)
   norm(Xg-Xp,'fro')/norm(Xp,'fro')

   norm(pd-pp)
   sum(Xp(nSDP,19:34))
   sum(Xg(nSDP,19:34))
end
end
function [G,p,a,g]=uparts(Xgr,nSDP,n_S)
   mdim  = nSDP*nSDP;
   G     = reshape(Xgr(1:mdim,1),nSDP,nSDP);
   p     = Xgr(mdim+1:mdim+n_S);
   a     = Xgr(mdim+n_S+1:mdim+2*n_S);
   g     = Xgr(mdim+2*n_S+1:mdim+3*n_S);
end
function [A_EC,b_EC,A_IC,s_IC,A_EV,b_EV,A_IV,s_IV,B_EV,B_IV,y_EC,y_IC,y_EV,y_IV,E_E,E_I] = getConstraints2(K,nSDP,n_l,n_u,n,initL,unlabeled,query,Yl,bSize,n_o,n_lbn)
%%
n_S       = n_l + n_u;     
n_q       = n_u;
dummy_pag = zeros(3*n,1);
%% Constraints in A_EC(u) = b_EC;
%A_EC      = sparse(nSDP*nSDP+3*n,2*n_l+n_u+2);%3*n_l+n_u+2
% b_EC      = zeros (2*n_l+n_u+2,1);
j         = 1;  
% cConstraint= [cConstraint,sum(q)==batchSize];% constraints on q 
% Constraint :1^T q = bSize
n_q       = numel(query);
q_ind     = [repmat(nSDP,n_q,1),query'];  % this is the indexes of q
R         = sparse([q_ind(:,1)',q_ind(:,2)'],[q_ind(:,2)',q_ind(:,1)'],repmat(0.5,2*n_q,1));
A_EC(:,j) = [reshape(R,nSDP*nSDP,1)',dummy_pag']; 
b_EC(j)   = bSize;
j         = j+1;
% cConstraint= [cConstraint,G_plus(nap+1,nap+1)==1];
% Constraint: G(nSDP,nSDP) = 1
R          = sparse(nSDP,nSDP,1,nSDP,nSDP);
A_EC(:,j)  = [reshape(R,nSDP*nSDP,1)',dummy_pag']; 
b_EC(j,1)  = 1;
j          = j+1;
% cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
%                  diag(G_plus(setunlab,setunlab))==r,...
% Constraint: diag(G_{ll,uu})==1-g, since g and a are removed it is
% equivalent to g=p+q , (i.e,diag(G_{ll,uu})==1-(p+q))
% this is equivalent to diag(G_{ll})+ p_l ==1
ap  = zeros(n_S,1);
for k=initL
    R         = sparse(k,k,1,nSDP,nSDP);
    ap(k)= 1;
    A_EC(:,j) = [reshape(R,nSDP*nSDP,1)',ap',zeros(2*n,1)']; 
    b_EC(j,1) = 1;
    ap(k)= 0;
    j = j+1;
end
% this is equivalent to diag(G_{uu})+ p_u + q==1
for k = 1:n_u
   ku        = unlabeled(k);
   kq        = query(k);
   R         = sparse([ku,kq,nSDP],[ku,nSDP,kq],[1,0.5,0.5],nSDP,nSDP);
   ap(ku)    = 1;
   A_EC(:,j) = [reshape(R,nSDP*nSDP,1)',ap',zeros(2*n,1)'];
   b_EC(j,1) = 1;
   ap(ku)    = 0;
   j = j+1; 
end
% cConstraint=[cConstraint, diag(G_plus(setQuery,setQuery))==q];
% Constraint: diag(G_qq)==q
for k = query
   R         = sparse([k,k,nSDP],[k,nSDP,k],[1,-0.5,-0.5],nSDP,nSDP);
   A_EC(:,j) = [reshape(R,nSDP*nSDP,1)',zeros(3*n,1)'];
   b_EC(j,1)   = 0;
   j = j+1; 
end
n_AEC    = j-1;
y_EC     = zeros(n_AEC,1);
%% Constraints in A_IC(u)<= b_IC
%A_IC = sparse(nSDP*nSDP+3*n,7*n_u+2+n_l);
%b_IC = zeros(7*n_u+2,1);
j = 1;
% cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
% Constraint: 1^T p <=n_o
ap(:)     = 1;
A_IC(:,j) = [zeros(nSDP*nSDP,1)',ap',zeros(2*n,1)'];
s_IC(j,1) = n_o;
ap(:)     = 0;
j = j+1;
% cConstraint=[cConstraint,sum(p)<=n_o]; %sum(p(initL))<=n_l*lnoiseper/100,
% Constraint: 1^T p_l <=n_lbn
assert(n_lbn<=n_o,'n_lbn, number of noisy labeled points must be less than or equal to n_o, number of noisy labeled points');
assert(n_lbn>0   ,'n_lbn, is zero and so constraint qualifications are not correct and dual problem is not equivalent to primal');
% ap(initL) = 1;
% A_IC(:,j) = [zeros(nSDP*nSDP,1)',ap',zeros(2*n,1)'];
% s_IC(j,1)   = n_lbn;
% ap(initL) = 0;
% j = j+1;
% constraint: p>=0
ap     = zeros(n,1);
p_ind  = unlabeled;
ag     = zeros(n,1);
for k=unlabeled
    ap(k) = -1;
    A_IC(:,j)    = [zeros(nSDP*nSDP,1)',ap',zeros(2*n,1)'];
    ap(k) = 0;
    s_IC(j,1)  = 0;
    j = j+1;
end
%cConstraint= [cConstraint,0<=q], where is q<=1? is hidden in p+q<=1,p>=0
% Constraint :q >=0 == -q <=0 
aa     = zeros(n,1);
ap     = zeros(n,1);
p_ind  = unlabeled;
ag     = zeros(n,1);
for k=query
    R = sparse([nSDP,k],[k,nSDP],[-0.5, -0.5],nSDP,nSDP);
    A_IC(:,j)    = [reshape(R,nSDP*nSDP,1)',zeros(3*n,1)'];
    s_IC(j,1)      = 0;
    j = j+1;
end
%cConstraint= [cConstraint,0<=q], where is q<=1? q<=1 is not necessary
%because of the constraint p+q <=1, all are positive. so, this constraint
%is deleted. 
% cConstraint= [cConstraint,r>=G_plus(setunlab,nap+1),r>=-G_plus(setunlab,nap+1)];
% Constraint: v <= a= 1-p-q :=> v+p+q <=1 
for k = 1:n_u
   ku        = unlabeled(k);
   kq        = query(k);
   R         = sparse([ku,nSDP,kq,nSDP],[nSDP,ku,nSDP,kq],[0.5,0.5,0.5,0.5],nSDP,nSDP);
   ap(ku)    = 1; 
   A_IC(:,j) = [reshape(R,nSDP*nSDP,1)',ap',zeros(2*n,1)'];
   s_IC(j,1) = 1;
   ap(ku)    = 0;
   j = j+1; 
end
%Constraint: -v - a<=0 :=>-v+p+q<=1
for k = 1:n_u
   ku        = unlabeled(k);
   kq        = query(k);
   R         = sparse([ku,nSDP,kq,nSDP],[nSDP,ku,nSDP,kq],[-0.5,-0.5,0.5,0.5],nSDP,nSDP);
   ap(ku)    = 1; 
   A_IC(:,j) = [reshape(R,nSDP*nSDP,1)',ap',zeros(2*n,1)'];
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
y_IC     = zeros(size(A_IC,2),1);
%% Constraints in A_EV(u) = b_EV + B_EV \beta
% constraint: v_l = y_l-\Phi(X_l)^T w_o
% b_EV = Yl-\Phi(X_l)^T w_o
%A_EV = sparse(nSDP*nSDP+3*n,n_l);
j=1;
for k = initL
   R = sparse([k,nSDP],[nSDP,k],[0.5,0.5],nSDP,nSDP);
   A_EV(:,j) = [reshape(R,nSDP*nSDP,1)',dummy_pag'];
   b_EV(j,1)   = Yl(k);
   j = j+1;
end
Iind     = speye(n_S,n_S);
I_l      = Iind(initL,1:n_S);
B_EV     = I_l*K;
y_EV     = zeros(size(A_EV,2),1);
n_AEV    = j-1;
%% Constraints in A_IV(u) <= b_IV
%A_IV = sparse(nSDP*nSDP+3*n,2*n_q+2*n_S);
j    = 1;
% Constraint: q <= 1+\phi(X)^T w_o
for k = query
    R = sparse([k,nSDP],[nSDP,k],[0.5,0.5],nSDP,nSDP);
    A_IV(:,j) = [reshape(R,nSDP*nSDP,1)',dummy_pag'];
    s_IV(j,1)   = 1;
    j = j + 1;
end
%Constraint: q <= 1-\phi(X)^T w_o
%A_IV  = [A_IV,A_IV];
for k = query
    R         = sparse([k,nSDP],[nSDP,k],[0.5,0.5],nSDP,nSDP);
    A_IV(:,j) = [reshape(R,nSDP*nSDP,1)',dummy_pag'];
    s_IV(j,1)   = 1;
    j         = j + 1;
end
% Constraint:  \phi(X)^T w_o  <= p
for k   = 1:n_S%unlabeled
   ap(k)    = -1;
   A_IV(:,j)= [zeros(nSDP*nSDP,1)',ap',zeros(2*n,1)'];
   s_IV(j,1)   = 0;
   ap(k)    = 0;
   j        = j+1;
end
% Constraint:  -\phi(X)^T w_o  - p <= 0 
for k=1:n_S%unlabeled
   ap(k)    = -1;
   A_IV(:,j)= [zeros(nSDP*nSDP,1)',ap',zeros(2*n,1)'];
   s_IV(j,1)  = 0;
   ap(k)    = 0;
   j        = j+1;
end

eye_q  = Iind(setdiff(1:n_S,initL),1:n_S);
I_rep  = [sparse(eye_q);-sparse(eye_q);speye(n_S);-speye(n_S)];
E_I    = [zeros(n_AIC,n_S);  I_rep];
E_E    = [zeros(n_AEC,n_S);  I_l  ];
B_IV   = I_rep*K;
y_IV   = zeros(size(A_IV,2),1);
n_AIV  = j-1;
%e12    = [ones(2*n_q,1);zeros(2*n_S,1)];
end
