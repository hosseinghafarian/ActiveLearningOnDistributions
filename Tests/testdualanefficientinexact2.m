function docode
close all
 
y = rand(10,1);
load('data4test','C_kM','nSDP','n_S','A_EC','b_EC','A_EV','b_EV');

G = C_kM;
Gr= [reshape(G,nSDP*nSDP,1);zeros(n_S,1);zeros(n_S,1);zeros(n_S,1)];
gprox = 0;


A_IC     = zeros(nSDP,nSDP);
A_IC(10,5)= 1;A_IC(5,10)=1; s_IC=G(5,10);

A_E  = A_EC';
A_EV = A_EV';
A_I  = [reshape(A_IC,1,nSDP*nSDP),zeros(1,n_S),zeros(1,n_S),zeros(1,n_S)];

X = sdpvar(nSDP,nSDP);
p = sdpvar(n_S,1);
a = sdpvar(n_S,1);
g = sdpvar(n_S,1);

Xr= [reshape(X,nSDP*nSDP,1);p;a;g];
s = sdpvar(1);
cConstraint = [X>=0,p>=0,a>=0,g>=0,A_E*Xr==b_EC,A_I*Xr==s,s<=s_IC,A_EV*Xr==b_EV];%
cObjective  = 1/2*norm(Xr-Gr)^2+1/2*(s-gprox)^2;
sol = optimize ( cConstraint,cObjective);
if sol.problem == 0 
    Xp = value(X);
    sp = value(s);
    pp = value(p);
    pa = value(a);
    pg = value(g);
end

n_E = size(b_EC,1);
n_EV= size(b_EV,1);
y_E = sdpvar(n_E,1);
y_EV= sdpvar(n_EV,1);
y_I = sdpvar(1);
S   = sdpvar(nSDP,nSDP);
pdu = sdpvar(n_S,1);
adu = sdpvar(n_S,1);
gdu = sdpvar(n_S,1);
v   = sdpvar(1);
DuXr= [reshape(S,nSDP*nSDP,1);pdu;adu;gdu];
cConstraint= [ S>=0,pdu>=0,adu>=0,gdu>=0,v>=0];
cObjective = -(b_EC'*y_E+b_EV'*y_EV)+ 1/2*norm(A_E'*y_E+A_I'*y_I+A_EV'*y_EV+DuXr+Gr)^2+1/2*norm(gprox+v-y_I)^2+v*s_IC;%
sol = optimize(cConstraint,cObjective);
if sol.problem==0 
   Sd = value(S);
   vd = value(v);
   y_Id   = value(y_I);
   y_Ed   = value(y_E); 
   y_EVd  = value(y_EV);
   A_yED  = value(A_E'*y_Ed);
   A_yEVD = value(A_EV'*y_EVd);
   y_Id   = value(y_I);
   A_yID  = value(A_I'*y_Id);
   Xgr    = Gr+value(DuXr)+A_E'*y_Ed+A_I'*y_Id+A_EV'*y_EVd;
   Xg     = reshape(Xgr(1:nSDP*nSDP,1),nSDP,nSDP);
   mdim   = nSDP*nSDP;
   pd     = Xgr(mdim+1:mdim+n_S);
   ad     = Xgr(mdim+n_S+1:mdim+2*n_S);
   gd     = Xgr(mdim+2*n_S+1:mdim+3*n_S);
   Xg2    = G+Sd+reshape(A_yED(1:nSDP*nSDP,1),nSDP,nSDP)+reshape(A_yID(1:nSDP*nSDP,1),nSDP,nSDP)+reshape(A_yEVD(1:nSDP*nSDP,1),nSDP,nSDP);
   norm(Xg-Xp,'fro')/norm(Xp,'fro')
   norm(Xg-Xp,'fro')/norm(Xp,'fro')
   norm(pd-pp)
   norm(pa-ad)
   norm(pg-gd)
end
end
function [G,p,a,g]=uparts(Xr,nSDP,n_S)
   mdim  = nSDP*nSDP;
   G     = reshape(Xr(1:mdim,1),nSDP,nSDP);
   p     = Xgr(mdim+1:mdim+n_S);
   a     = Xgr(mdim+n_S+1:mdim+2*n_S);
   g     = Xgr(mdim+2*n_S+1:mdim+3*n_S);
end
