function [x_0, dualvars_0, gscale, scaledoperator] = scaleProblem(learningparams,operators,x_0, dualvars_0)

global g_x_gscale

scaledoperator    = operators;
%% scaling data
sscale       = sqrt(norm(x_0.st,'fro'));
if sscale   ==0, sscale =1; end
betascale = sqrt(norm(x_0.w_obeta,'fro'));
if betascale==0, betascale =1; end
uscale    = sqrt(norm(x_0.u,'fro'));
if uscale   ==0, uscale =1; end

gscale           = max(uscale,max(sscale,betascale));
x_0.u            = x_0.u           / gscale;
x_0.w_obeta      = x_0.w_obeta     / gscale;
x_0.st           = x_0.st          / gscale;
dualvars_0.y_EC  = dualvars_0.y_EC / gscale;
dualvars_0.y_EV  = dualvars_0.y_EV / gscale;
dualvars_0.y_IC  = dualvars_0.y_IC / gscale;
dualvars_0.y_IV  = dualvars_0.y_IV / gscale;
dualvars_0.S     = dualvars_0.S    / gscale;
dualvars_0.Z     = dualvars_0.Z    / gscale;
dualvars_0.v     = dualvars_0.v    / gscale;

scaledoperator.b_EC         = scaledoperator.b_EC        /gscale; % based on the constraints, when Ghat.u is scaled, b_EC,b_EV,s_IC,s_IV is also must be scaled
scaledoperator.b_EV         = scaledoperator.b_EV        /gscale;
scaledoperator.s_IC         = scaledoperator.s_IC        /gscale;
scaledoperator.s_IV         = scaledoperator.s_IV        /gscale;

g_x_gscale   = gscale;

normA_EC     = norm(scaledoperator.A_EC  ,'fro');
normA_EV     = norm(scaledoperator.A_EV  ,'fro');
normA_IC     = norm(scaledoperator.A_IC  ,'fro');
normA_IV     = norm(scaledoperator.A_IV  ,'fro');
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
%normBV      = max(norm(B_EV,'fro'),norm(B_IV,'fro'));
%betascaling: \hat{\beta}= \sqrt(lambda_o) \beta. therefore, \hat{B}_EV
%=B_EV/(\sqrt(lambda_o)) and \hat{B}_IV =B_IV/(\sqrt(lambda_o)) 
% operat.B_EV = B_EV/normBV;
% operat.B_IV = B_IV/normBV;
% if maxEscale > maxIscale %&& maxEscale> maxCscale 
%     scaleA_IC = maxIscale/maxEscale;
% else%if maxIscale > maxEscale&& maxIscale > maxCscale
%     scaleA_EC = maxEscale/maxIscale;
% %     scaleA_IV = maxEscale/maxIscale;
% end
end
