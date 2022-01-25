function docode
close all
recomp=1;
y = rand(10,1);

load('data4test','lambda','KE','alphapre','alphatild','nSDP','n_S','A_EC','b_EC','A_IC','s_IC','A_EV','b_EV','A_IV','s_IV','B_EV','B_IV');
alphanorm  = norm(alphapre);
n_u        = nSDP-n_S-1;
%alphapre   = alphapre/alphanorm;

c_a   =2;
C_kM          = [(alphapre*alphapre').*KE/(2*lambda),zeros(nSDP-1,1);zeros(1,nSDP-1),0];
alphatild     = alphapre(1:n_S);

cp   = zeros(n_S,1);
ca   = -c_a*ones(n_S,1);
cg   = alphatild;
c_k  = [reshape(C_kM,nSDP*nSDP,1);cp;ca;cg];%first part for C_kM, the other parts for pag: p,a,g

n_IC = 2;%size(s_IC,1);
n_IV = 2*n_S;%2*n_S+
query = n_S+1:nSDP-1;
A_ICcopy = A_IC(:,1:n_IC);
A_IC     = A_ICcopy';
s_IC = s_IC(1:n_IC);
A_IVcopy = A_IV(:,1:n_IV);
A_IV     = A_IVcopy';
s_IV = s_IV(1:n_IV);
A_EC = A_EC';
A_EV = A_EV';
%% scaling data
normA_EC = norm(A_EC,'fro');
normA_EV = norm(A_EV,'fro');
normA_IC = norm(A_IC,'fro');
normC_kM = norm(C_kM,'fro');
normcp   = norm(cp);
normca   = norm(ca);
normcg   = norm(cg);

maxCscale= max(max(max(normC_kM,normcp),normcg),normca);
scaleA_EC = 1;
scaleA_EV = 1;
scaleA_IC = 1;
scaleC_kM = 1;
scalecp   = 1;
scaleca   = 1;
scalecg   = 1;

maxEscale = max(normA_EC,normA_EV);
maxIscale = normA_IC;

if maxEscale > maxIscale %&& maxEscale> maxCscale 
    scaleA_IC = maxIscale/maxEscale;
%     scaleC_kM = scaleC_kM/maxEscale;
%     scalecp   = scalecp / maxEscale;
%     scaleca   = scaleca / maxEscale;
else%if maxIscale > maxEscale&& maxIscale > maxCscale
    scaleA_EC = maxEscale/maxIscale;
    scaleA_IV = maxEscale/maxIscale;
%     scaleC_kM = scaleC_kM/maxIscale;
%     scalecp   = scalecp / maxIscale;
%     scaleca   = scaleca / maxIscale;
% elseif maxCscale > maxEscale && maxCscale > maxIscale
%     scaleA_EC = maxEscale/maxCscale;
%     scaleA_IV = maxEscale/maxCscale;
%     scaleA_IC = maxIscale/maxCscale;
%     scaleC_kM = max(scaleC_kM,normC_kM)/maxCscale;
%     scalecp   = max(scalecp,normcp) / maxCscale;
%     scaleca   = max(scaleca,normca) / maxCscale;
end
%G = G/scaleG;
Gr  = [reshape(C_kM/scaleC_kM,nSDP*nSDP,1);cp/scalecp;ca/scaleca;cg/scalecg;];

n_EC = size(b_EC,1);
n_EV = size(b_EV,1);
scaleG = norm(Gr,'fro');
gprox= zeros(n_IC+n_IV,1);

A_EC = A_EC/scaleA_EC;
b_EC = b_EC/scaleA_EC;
A_EV = A_EV/scaleA_EV;
b_EV = b_EV/scaleA_EV;
A_IC     = A_IC/scaleA_IC;
s_IC     = s_IC/scaleA_IC;

%A_I  = [reshape(A_IC,1,nSDP*nSDP),zeros(1,n_S),zeros(1,n_S),zeros(1,n_S)];

X = sdpvar(nSDP,nSDP);
p = sdpvar(n_S,1);
a = sdpvar(n_S,1);
g = sdpvar(n_S,1);

Xr= [reshape(X,nSDP*nSDP,1);p;a;g];
s1 = sdpvar(n_IC,1);
s2 = sdpvar(n_IV,1);
s  = [s1;s2];
cConstraint = [X>=0,p>=0,a>=0,g>=0,X(nSDP,query)>=0,A_EC*Xr==b_EC,A_IC*Xr==s1,A_IV*Xr==s2,A_EV*Xr==b_EV,s1<=s_IC,s2<=s_IV];%
cObjective  = 1/2*norm(Xr-Gr)^2+1/2*norm(s-gprox)^2;
if recomp==1
    sol = optimize ( cConstraint,cObjective);
    if sol.problem == 0 
        pobj   = value(cObjective);
        Xp     = value(X)/scaleC_kM;
        sp     = value(s);
        sp2    = value(s2);
        pp     = value(p)/scalecp;
        pa     = value(a)/scaleca;
        pg     = value(g)/scalecg;
        fvpAEC = A_EC*value(Xr)-b_EC;
        Sdup   = dual(cConstraint(1));
        pdup   = dual(cConstraint(2));
        adup   = dual(cConstraint(3));
        gdup   = dual(cConstraint(4));
        Xdup   = dual(cConstraint(5));
        AECdup = dual(cConstraint(6));
        AICdup = dual(cConstraint(7));
        AIVdup = dual(cConstraint(8));
        AEVdup = dual(cConstraint(9));
        save('Pvals','pobj','Xp','sp','pp','pa','pg');
    end
else
    load('Pvals','pobj','Xp','sp','pp','pa','pg');
end

y_EC = sdpvar(n_EC,1);
y_EV = sdpvar(n_EV,1);
y_IC = sdpvar(n_IC,1);
y_IV = sdpvar(n_IV,1);
S    = sdpvar(nSDP,nSDP);
pdu  = sdpvar(n_S,1);
adu  = sdpvar(n_S,1);
gdu  = sdpvar(n_S,1);
Z    = sdpvar(nSDP,nSDP);
zp   = sdpvar(n_S,1);
za   = sdpvar(n_S,1);
zg   = sdpvar(n_S,1);
Zr   = [reshape(Z,nSDP*nSDP,1);zeros(3*n_S,1)]; %
v    = sdpvar(n_IC+n_IV,1);
y_I  = [y_IC;y_IV];
s_I  = [s_IC;s_IV];
DuXr = [reshape(S,nSDP*nSDP,1);pdu;adu;gdu];
% Xgt    = Gr+value(DuXr)+A_EC'*y_EC/scaleA_EC+A_IC'*y_IC/scaleA_IC+A_EV'*y_EV/scaleA_EV;
% XgM     = reshape(Xgt(1:nSDP*nSDP,1),nSDP,nSDP);
cConstraint= [ S>=0,pdu>=0,adu>=0,gdu>=0,v>=0];%,Z(1:n_S,:)==0,Z(:,1:n_S)==0,Z(n_S+1:nSDP,n_S+1:nSDP)>=0];%,Z(nSDP,query)>=0];
cObjective = -(b_EC'*y_EC+b_EV'*y_EV)+ 1/2*norm(A_EC'*y_EC+A_IC'*y_IC+A_EV'*y_EV+A_IV'*y_IV+DuXr+Gr)^2+1/2*norm(gprox+v-y_I)^2+v'*s_I-1/2*norm(Gr,'fro')^2-1/2*norm(gprox)^2;%
sol = optimize(cConstraint,cObjective);
% if sol.problem==0 
   dobj = value(cObjective);
   Sd = value(S);
   vd = value(v);
   Zd     = value(Z);
   y_ICd  = value(y_IC);
   y_ECd  = value(y_EC); 
   y_EVd  = value(y_EV);
   A_yECD = value(A_EC'*y_ECd);
   A_yEVD = value(A_EV'*y_EVd);
   A_yICD = value(A_IC'*y_ICd);
   
   %Xgr   = Gr+value(DuXr)+scaleA_EC*A_EC'*y_ECd+scaleA_IC*A_IC'*y_ICd+scaleA_EV*A_EV'*y_EVd;
   Xgr    = Gr+value(DuXr)+A_EC'*y_ECd/scaleA_EC+A_IC'*y_ICd/scaleA_IC+A_EV'*y_EVd/scaleA_EV;
   feasAEC= norm(A_EC*Xgr-b_EC);
   fvAEC  = A_EC*Xgr-b_EC;
   feasAIC= norm(max(s_IC-A_IC*Xgr,0));
   feasAEV= norm(A_EV*Xgr-b_EV);
   
   Xg     = reshape(Xgr(1:nSDP*nSDP,1),nSDP,nSDP);
   mdim   = nSDP*nSDP;
   pd     = Xgr(mdim+1:mdim+n_S);
   ad     = Xgr(mdim+n_S+1:mdim+2*n_S);
   gd     = Xgr(mdim+2*n_S+1:mdim+3*n_S);
   %Xg2    = G+Sd+reshape(A_yECD(1:nSDP*nSDP,1),nSDP,nSDP)+reshape(A_yICD(1:nSDP*nSDP,1),nSDP,nSDP)+reshape(A_yEVD(1:nSDP*nSDP,1),nSDP,nSDP);
    trace(Sdup*Xg)
    norm(pdu-pdup)
    norm(adu-adup)
    norm(gdu-gdup)
    % Xdup   = dual(cConstraint(5));
    norm(AECdup-y_EC)
    norm(AICdup-y_IC)
    norm(AIVdup-y_IV)
    norm(AEVdup-y_EV)
   norm(Gr,'fro')
   norm(value(DuXr))
   norm(A_EC'*y_ECd/scaleA_EC)
   norm(A_IC'*y_ICd/scaleA_IC)
   norm(A_EV'*y_EVd/scaleA_EV)
   norm(Xg-Xp,'fro')/norm(Xp,'fro')
   %norm(Xg2-Xp,'fro')/norm(Xp,'fro')
   norm(pd-pp)
   norm(pa-ad)
   norm(pg-gd)
   sum(Xp(nSDP,19:34))
   sum(Xg(nSDP,19:34))
% end
end
function [G,p,a,g]=uparts(Xgr,nSDP,n_S)
   mdim  = nSDP*nSDP;
   G     = reshape(Xgr(1:mdim,1),nSDP,nSDP);
   p     = Xgr(mdim+1:mdim+n_S);
   a     = Xgr(mdim+n_S+1:mdim+2*n_S);
   g     = Xgr(mdim+2*n_S+1:mdim+3*n_S);
end
