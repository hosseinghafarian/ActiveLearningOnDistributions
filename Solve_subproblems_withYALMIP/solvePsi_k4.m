function [x_k,dualvars_k, comp, feas, eqIneq] = solvePsi_k4(objectivefunc,x_0, g_acc_x, beta_kx,operators,learningparams,optparams)
%% Setting Proximal parameters 
tolgap = 10^-4;
%setting proximal x^k for now, just fot the test
global cnstData;

Ghat.u       = (1/beta_kx)*(-               g_acc_x.st      + beta_kx*x_0.u      );
Ghat.w_obeta = (1/beta_kx)*(- cnstData.Qinv*g_acc_x.w_obeta + beta_kx*x_0.w_obeta);
Ghat.st      = (1/beta_kx)*(-               g_acc_x.st      + beta_kx*x_0.st     );
objectivefunc_primal = objectivefunc.primal;
objectivefunc_dual   = objectivefunc.dual; 
[gscale,scoperators]                      = scaleProblem(learningparams,operators,Ghat);

[x_k, obj_val_primal, sol_problem_primal] = solve_primal(objectivefunc_primal,Ghat,scoperators,learningparams,optparams);
[x_k] = descale_in_primal(gscale, learningparams, operators,x_k);

%% Dual problem 
[dualvars_k, obj_val_dual, sol_problem_dual] = solve_dual(objectivefunc_dual,Ghat,scoperators,learningparams,optparams);
[dualvars_k ] = descale_in_dual(gscale, learningparams, scoperators, dualvars_k);

if abs(obj_val_primal-obj_val_dual)/(1+abs(obj_val_primal)) > tolgap, assert(true,'gap is not zero in function solvePsi_k4'),end
%% Check primal-dual pair
[x_k_from_dual ]                           = x_conv_from_dual(dualvars_k, Ghat, scoperators);
[comp.yAEC, comp.yAEV,comp.yAIC,comp.yAIV,...
          comp.SDP , comp.V   ,...
          feas.AEC,feas.AIC,feas.AEV,feas.AIV] = checkComplementarity(scoperators, x_k, dualvars_k);
[eqIneq.equalitydiff, eqIneq.eq, eqIneq.Inequality, eqIneq.Ineq]       = ConstraintsCheck(x_k, learningparams);

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

