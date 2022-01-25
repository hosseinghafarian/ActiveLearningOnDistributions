function [v_k_x,dualvars]=solvePsi_k2(x_k,operators,dualvars,...
                            x_G,c_u,c_beta,c_s,accumfgrad,learningparams,optparams,arho,mu_reg)
%% Setting Proximal parameters 
%setting proximal x^k for now, just fot the test
global cnstData;

gamma_reg = arho+mu_reg; 
Ghat.u    = (1/gamma_reg)*(c_u    + arho*x_G.u      + mu_reg*x_k.u);
Ghat.w_obeta = (1/gamma_reg)*(cnstData.Qinv*c_beta + arho*x_G.w_obeta+ mu_reg*x_k.w_obeta);
Ghat.st   = (1/gamma_reg)*(c_s    + arho*x_G.st     + mu_reg*x_k.st);

[gscale,A_EC,A_EV,A_IC,A_IV,B_EV,B_IV,b_EC,b_EV,s_IC,s_IV,scaledoperator] = scaleProblem(operators,Ghat);
%% Primal problem 
n_IC        = size(A_IC,1);
n_IV        = size(A_IV,1);
X           = sdpvar(cnstData.nSDP,cnstData.nSDP);
p           = sdpvar(cnstData.n_S,1);
w_obeta     = sdpvar(cnstData.n_S,1);
Xr          = [reshape(X,cnstData.nSDP*cnstData.nSDP,1);p;];
s1          = sdpvar(n_IC,1);
s2          = sdpvar(n_IV,1);
s           = [s1;s2];
iAEC        = 3; iAEV        = 6; iAIC        = 4; iAIV        = 5;
pcConstraint= [X>=0,p>=0,A_EC*Xr==b_EC,A_IC*Xr==s1,A_IV*Xr==s2-B_IV'*w_obeta,A_EV*Xr==b_EV-B_EV'*w_obeta,s1<=s_IC,s2<=s_IV, ...
               X(cnstData.extendInd,cnstData.nSDP)>=0, X(cnstData.nSDP,cnstData.extendInd)>=0 ];
pcObjective  = 1/2*norm(Xr-Ghat.u)^2+1/2*norm(s-Ghat.st)^2+1/2*(w_obeta-Ghat.w_obeta)'*cnstData.Q*(w_obeta-Ghat.w_obeta);
sol = optimize ( pcConstraint, pcObjective,sdpsettings('dualize',0));
if sol.problem == 0 
    objp   = value(pcObjective);
    Xrp    = value(Xr);
    qp     = value(X(cnstData.extendInd,cnstData.nSDP));
    pp     = value(p);
    wobetap= value(w_obeta);
    sp     = [value(s1);value(s2)];
end
%% Dual problem 
n_EC    = size(A_EC,1); n_EV = size(A_EV,1);
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
Aysdp   = A_EC'*y_EC+A_IC'*y_IC+A_EV'*y_EV+A_IV'*y_IV;
Bysdp   = B_EV*y_EV+B_IV*y_IV;
y_I     = [y_IC;y_IV];
s_I     = [s_IC;s_IV];

dcConstraint= [ Stil>=0,pdu>=0,v>=0,Zpartq>=0];
dcObjective = (b_EC'*y_EC+b_EV'*y_EV)- 1/2*norm(Aysdp+S+Z+Ghat.u)^2 ...
             - 1/2*(Bysdp+cnstData.Q*Ghat.w_obeta)'*cnstData.Qinv*(Bysdp+cnstData.Q*Ghat.w_obeta)...
             -1/2*norm(v+y_I-Ghat.st)^2-v'*s_I+const2;
dcObjective = - dcObjective;
sol = optimize(dcConstraint,dcObjective,sdpsettings('dualize',0));
if sol.problem==0 
   objd     = value(dcObjective);
   objequal = abs(objd+objp);
   Sd       = value(S);
   vd       = value(v);
   Zd       = value(Z);
   Zpartqd  = value(Zpartq);
   y_ICd    = value(y_IC);
   y_IVd    = value(y_IV);
   y_ECd    = value(y_EC); 
   y_EVd    = value(y_EV);
   Xrd      = Ghat.u + A_EC'*y_ECd+A_IC'*y_ICd+A_EV'*y_EVd +A_IV'*y_IVd+ value(S)+ value(Z);
   Xd       = reshape(Xrd(1:cnstData.nSDP*cnstData.nSDP),cnstData.nSDP,cnstData.nSDP);
   qd       = Xd(cnstData.extendInd,cnstData.nSDP);
   wobetad  = Ghat.w_obeta+cnstData.Qinv*(B_EV*y_EVd+B_IV*y_IVd);
   sd       = Ghat.st-(vd+value(y_I));
else
    assert('1==1','Problem is not successfull solved');
end
%% Check primal-dual pair
[compSDP,...
         compAEC,compAEV,compAIC,compAIV,compV,compQ,...
         feasAEC,feasAIC,feasAEV,feasAIV] = checkComplementarity(scaledoperator,pcConstraint,Xrd,Xrp,wobetad,sd, dcConstraint, y_ECd,y_EVd,y_ICd,y_IVd,Sd,Zpartqd,vd,qd,n_IC,n_IV);
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
   fvpAEV = operat.b_EV-operat.A_EV*Xrd-operat.B_EV'*wobetad;
   compyAEV = fvpAEV'*y_EVd;  
   fvpAIC = sd(1:n_IC)-operat.A_IC*Xrd;
   compyAIC = fvpAIC'*y_ICd;
   fvpAIV = sd(n_IC+1:end)-operat.A_IV*Xrd-operat.B_IV'*wobetad;
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
function [gscale,A_EC,A_EV,A_IC,A_IV,B_EV,B_IV,b_EC,b_EV,s_IC,s_IV,operat] = scaleProblem(operators,Ghat)

A_EC      = operators.A_EC;
b_EC      = operators.b_EC;
A_IC      = operators.A_IC;
s_IC      = operators.s_IC;
A_EV      = operators.A_EV;
b_EV      = operators.b_EV;
A_IV      = operators.A_IV;
s_IV      = operators.s_IV;
B_EV      = operators.B_EV;
B_IV      = operators.B_IV;

n_IC = size(s_IC,1);
n_IV = size(s_IV,1);%2*n_S;%2*n_S+
A_IC = A_IC';
s_IC = s_IC(1:n_IC);
A_IV = A_IV';
s_IV = s_IV(1:n_IV);
A_EC = A_EC';
A_EV = A_EV';
%% scaling data
sscale    = sqrt(norm(Ghat.st,'fro'));
if sscale    ==0, sscale =1; end
betascale = sqrt(norm(Ghat.w_obeta,'fro'));
if betascale ==0, betascale =1; end
uscale    = sqrt(norm(Ghat.u,'fro'));
if uscale    ==0, uscale =1; end

gscale       = max(uscale,max(sscale,betascale));
Ghat.u       = Ghat.u      /gscale;
Ghat.w_obeta = Ghat.w_obeta/gscale;
Ghat.st      = Ghat.st     /gscale;
b_EC         = b_EC        /gscale; % based on the constraints, when Ghat.u is scaled, b_EC,b_EV,s_IC,s_IV is also must be scaled
b_EV         = b_EV        /gscale;
s_IC         = s_IC        /gscale;
s_IV         = s_IV        /gscale;

normA_EC     = norm(A_EC  ,'fro');
normA_EV     = norm(A_EV  ,'fro');
normA_IC     = norm(A_IC  ,'fro');
normA_IV     = norm(A_IV  ,'fro');
scaleA_EC    = 1;
scaleA_EV    = 1;
scaleA_IC    = 1;
scaleA_IV    = 1;
scalec_k     = 1;
scalecp      = 1;
scaleca      = 1;
scalecg      = 1;
scaleG       = 1;%norm(Gr,'fro');
maxEscale    = max(normA_EC,normA_EV);
maxIscale    = max(normA_IC,normA_IV);
A_EC        = A_EC/scaleA_EC;
b_EC        = b_EC/scaleA_EC;
A_EV        = A_EV/scaleA_EV;
b_EV        = b_EV/scaleA_EV;
A_IC        = A_IC/scaleA_IC;
s_IC        = s_IC/scaleA_IC;
operat.A_EC = A_EC  ;
operat.b_EC = b_EC   ;
operat.A_IC = A_IC   ;
operat.s_IC = s_IC;
operat.A_EV = A_EV;
operat.b_EV = b_EV;
operat.A_IV = A_IV;
operat.s_IV = s_IV;
operat.B_EV = B_EV;
operat.B_IV = B_IV;

% if maxEscale > maxIscale %&& maxEscale> maxCscale 
%     scaleA_IC = maxIscale/maxEscale;
% else%if maxIscale > maxEscale&& maxIscale > maxCscale
%     scaleA_EC = maxEscale/maxIscale;
% %     scaleA_IV = maxEscale/maxIscale;
% end
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
