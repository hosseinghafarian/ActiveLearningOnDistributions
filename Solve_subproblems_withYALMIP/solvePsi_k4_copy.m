function [v_k_x,dualvars,comp, feas, eqIneq]=solvePsi_k4_copy(objectivefunc,x_0,g_acc_x,beta_kx,operators,learningparams,optparams)
%% Setting Proximal parameters 
%setting proximal x^k for now, just fot the test
global cnstData;
comp=0; feas=0; eqIneq=0;

Ghat.u    = (1/beta_kx)   *(- g_acc_x.u                     + beta_kx*x_0.u      );
Ghat.w_obeta = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_0.w_obeta);
Ghat.st   = (1/beta_kx)   *(- g_acc_x.st                    + beta_kx*x_0.st     );

[gscale,scoperator] = scaleProblem(learningparams,operators,Ghat);
%% Primal problem 
n_IC        = scoperator.n_AIC;
n_IV        = scoperator.n_AIV;
X           = sdpvar(cnstData.nSDP,cnstData.nSDP);
p           = sdpvar(cnstData.n_S,1);
w_obeta     = sdpvar(cnstData.n_S,1);
Xr          = [reshape(X,cnstData.nSDP*cnstData.nSDP,1);p;];
s1          = sdpvar(n_IC,1);
s2          = sdpvar(n_IV,1);
s           = [s1;s2];
iAEC        = 3; iAEV        = 6; iAIC        = 4; iAIV        = 5;
pcConstraint= [X>=0,p>=0,scoperator.A_EC*Xr==scoperator.b_EC,scoperator.A_IC*Xr==s1,scoperator.A_IV*Xr+scoperator.B_IV*w_obeta==s2,...
               scoperator.A_EV*Xr+scoperator.B_EV*w_obeta==scoperator.b_EV,s1<=scoperator.s_IC,s2<=scoperator.s_IV, ...
               X(cnstData.extendInd,cnstData.nSDP)>=0, X(cnstData.nSDP,cnstData.extendInd)>=0 ];
pcObjective  = 1/2*norm(Xr-Ghat.u)^2+1/2*norm(s-Ghat.st)^2+1/2*(w_obeta-Ghat.w_obeta)'*cnstData.Q*(w_obeta-Ghat.w_obeta);
sol = optimize ( pcConstraint, pcObjective,sdpsettings('dualize',0,'verbose','0'));
objp   = value(pcObjective);
Xrp    = value(Xr);
qp     = value(X(cnstData.extendInd,cnstData.nSDP));
pp     = value(p);
wobetap= value(w_obeta);
sp     = [value(s1);value(s2)];
% if sol.problem == 0 
%     objp   = value(pcObjective);
%     Xrp    = value(Xr);
%     qp     = value(X(cnstData.extendInd,cnstData.nSDP));
%     pp     = value(p);
%     wobetap= value(w_obeta);
%     sp     = [value(s1);value(s2)];
% else
%     assert(true,'Cannot solve problem');
% end
v_k_x.u         = gscale*Xrp;
v_k_x.st        = gscale*sp;
v_k_x.w_obeta   = gscale*wobetap; 
dualvars.y_EC   = 0;%y_ECd;
dualvars.y_EV   = 0;%y_EVd;
dualvars.y_IC   = 0;%y_ICd;
dualvars.y_IV   = 0;%y_IVd;
dualvars.S      = 0;%Sd;%%%%%%%%%%%%%%%%%%%%

%% Dual problem 
n_EC    = size(scoperator.A_EC,1); n_EV = size(scoperator.A_EV,1);
const2  = 1/2* compNorm(Ghat,cnstData.Q);
y_EC    = sdpvar(n_EC,1); y_IC    = sdpvar(n_IC,1);
y_EV    = sdpvar(n_EV,1); y_IV    = sdpvar(n_IV,1);
Stil    = sdpvar(cnstData.nSDP,cnstData.nSDP);
pdu     = sdpvar(cnstData.n_S,1);
Zpartq  = sdpvar(cnstData.n_u,1);
v       = sdpvar(n_IC+n_IV,1);
zapp    = [zeros(cnstData.n_S,1);Zpartq];
ZMat    = [zeros(cnstData.nSDP-1,cnstData.nSDP-1),zapp;zapp',0];
Z       = [reshape(ZMat,cnstData.nSDP*cnstData.nSDP,1);zeros(cnstData.n_S,1)]; 
S       = [reshape(Stil,cnstData.nSDP*cnstData.nSDP,1);pdu];
Aysdp   = scoperator.A_EC'*y_EC+scoperator.A_IC'*y_IC+scoperator.A_EV'*y_EV+scoperator.A_IV'*y_IV;
Bysdp   = scoperator.B_EV'*y_EV+scoperator.B_IV'*y_IV;
y_I     = [y_IC;y_IV];
s_I     = [scoperator.s_IC;scoperator.s_IV];

dcConstraint= [ Stil>=0,pdu>=0,v>=0,Zpartq>=0];
dcObjective = (scoperator.b_EC'*y_EC+scoperator.b_EV'*y_EV)- 1/2*norm(Aysdp+S+Z+Ghat.u)^2 ...
             - 1/2*(Bysdp+cnstData.Q*Ghat.w_obeta)'*cnstData.Qinv*(Bysdp+cnstData.Q*Ghat.w_obeta)...
             -1/2*norm(v+y_I-Ghat.st)^2-v'*s_I+const2;
dcObjective = - dcObjective;
sol = optimize(dcConstraint,dcObjective,sdpsettings('dualize',0,'verbose','0'));

objd     = value(dcObjective);
objequal = abs(objd+objp);
if(objequal/abs(objd) >0.1)
    assert(true,'gap is not zero');
end 
Sd       = value(S);
vd       = value(v);
Zd       = value(Z);
Zpartqd  = value(Zpartq);
y_ICd    = value(y_IC);
y_IVd    = value(y_IV);
y_ECd    = value(y_EC); 
y_EVd    = value(y_EV);
Xrd      = Ghat.u + scoperator.A_EC'*y_ECd+scoperator.A_IC'*y_ICd+scoperator.A_EV'*y_EVd +scoperator.A_IV'*y_IVd+ value(S)+ value(Z);
Xd       = reshape(Xrd(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
qd       = Xd(cnstData.extendInd,cnstData.nSDP);
wobetad  = Ghat.w_obeta+cnstData.Qinv*(scoperator.B_EV'*y_EVd+scoperator.B_IV'*y_IVd);
sd       = Ghat.st-(vd+value(y_I));
%% Check primal-dual pair
[compSDP,...
         compAEC,compAEV,compAIC,compAIV,compV,compQ,...
         feasAEC,feasAIC,feasAEV,feasAIV] = checkComplementarity(scoperator,pcConstraint,Xrd,Xrp,wobetad,sd, dcConstraint, y_ECd,y_EVd,y_ICd,y_IVd,Sd,Zpartqd,vd,qd,n_IC,n_IV);
[gap,uDiff,woDiff,stDiff] = checkGap_VarsEquality(objp,objd,Xrp,Xrd,wobetap,wobetad,sp,sd);
[ecrDiff,evrDiff,icrDiff,ivrDiff ] = checkDualVars(pcConstraint,y_ECd,iAEC,y_ICd,iAIC,y_IVd,iAIV,y_EVd,iAEV);
epsvalue = 0.01;
assert(gap/(1+abs(objp))<epsvalue);
v_k_x.u         = gscale*Xrd;
v_k_x.st        = gscale*sd;
v_k_x.w_obeta   = gscale*wobetad; 
dualvars.y_EC   = y_ECd;
dualvars.y_EV   = y_EVd;
dualvars.y_IC   = y_ICd;
dualvars.y_IV   = y_IVd;
dualvars.S      = Sd;%%%%%%%%%%%%%%%%%%%%
end
function [compSDP,...
          compyAEC,compyAEV,compyAIC,compyAIV,compV,compQ,...
          feasAEC,feasAIC,feasAEV,feasAIV] = checkComplementarity(operat,pcConstraint,Xrp,Xrd,wobetad,sd, dcConstraint, y_ECd,y_EVd,y_ICd,y_IVd,Sd,Zpartqd,vd,qd,n_IC,n_IV)
   global cnstData
   s_I     = [operat.s_IC;operat.s_IV];
   compSDP = trace(Sd'*Xrp);
   %pdup   = dual(pcConstraint(2));
   %compP  = p'*pdup;
   fvpAEC = operat.b_EC-operat.A_EC*Xrd;
   compyAEC = fvpAEC'*y_ECd;
   fvpAEV = operat.b_EV-operat.A_EV*Xrd-operat.B_EV*wobetad;
   compyAEV = fvpAEV'*y_EVd;  
   fvpAIC = sd(1:n_IC)-operat.A_IC*Xrd;
   compyAIC = fvpAIC'*y_ICd;
   fvpAIV = sd(n_IC+1:end)-operat.A_IV*Xrd-operat.B_IV*wobetad;
   compyAIV = fvpAIV'*y_IVd;
   vdup     = sd-s_I;
   compV  = vdup'*vd;
   compQ  = Zpartqd'*qd;
   
   feasAEC= norm(compyAEC);
   feasAIV= norm(compyAIV);
   feasAEV= norm(compyAEV);
   feasAIC= norm(compyAIC);
end
function [gap,uDiff,woDiff,stDiff] = checkGap_VarsEquality(objp,objd,Xrp,Xrd,wobetap,wobetad,sp,sd)

   gap      = abs(objp+objd);
   uDiff    = norm(Xrp-Xrd)/norm(Xrp);
   woDiff   = norm(wobetap-wobetad)/norm(wobetap);
   nsp      = norm(sp);
   if nsp==0, nsp = 1; end
   stDiff   = norm(sp-sd)/nsp;
end
function [ecrDiff,evrDiff,icrDiff,ivrDiff ] = checkDualVars(pcConstraint,y_ECd,iAEC,y_ICd,iAIC,y_IVd,iAIV,y_EVd,iAEV)
   AECdup = -dual(pcConstraint(iAEC));
   AEVdup = -dual(pcConstraint(iAEV));
   AICdup = -dual(pcConstraint(iAIC));
   AIVdup = -dual(pcConstraint(iAIV));
   ecrDiff = norm(AECdup-y_ECd)/norm(y_ECd);
   evrDiff = norm(AEVdup-y_EVd)/norm(y_EVd);
   icrDiff = norm(AICdup-y_ICd)/norm(y_ICd);
   ivrDiff = norm(AIVdup-y_IVd)/norm(y_IVd);
end
function [y_ECtil,y_EVtil,y_ICtil,y_IVtil,Stil] = loaddualvars(dualvarsPre)
        y_ECtil  = dualvarsPre.y_EC ;
        y_EVtil  = dualvarsPre.y_EV ;
        y_ICtil  = dualvarsPre.y_IC ;
        y_IVtil  = dualvarsPre.y_IV ;
        Stil     = dualvarsPre.S    ; 
    end
function [G,p,a,g]=uparts(Xgr,nSDP,n_S)
   mdim  = nSDP*nSDP;
   G     = reshape(Xgr(1:mdim,1),nSDP,nSDP);
   p     = Xgr(mdim+1:mdim+n_S);
   a     = Xgr(mdim+n_S+1:mdim+2*n_S);
   g     = Xgr(mdim+2*n_S+1:mdim+3*n_S);
end
function value = compNorm(x_k,Q)
    value = norm(x_k.u)^2+norm(x_k.st)^2+x_k.w_obeta'*Q*x_k.w_obeta;
end
