function [dualvars_0 ] = descale_in_dual(gscale, learningparams, operators, dualvars_0)

%descaledoperator    = operators;
%% descaling data
dualvars_0.y_EC  = dualvars_0.y_EC * gscale;
dualvars_0.y_EV  = dualvars_0.y_EV * gscale;
dualvars_0.y_IC  = dualvars_0.y_IC * gscale;
dualvars_0.y_IV  = dualvars_0.y_IV * gscale;
dualvars_0.S     = dualvars_0.S    * gscale;
dualvars_0.Z     = dualvars_0.Z    * gscale;
dualvars_0.v     = dualvars_0.v    * gscale;

% descaledoperator.b_EC  = descaledoperator.b_EC  *gscale; % based on the constraints, when Ghat.u is scaled, b_EC,b_EV,s_IC,s_IV is also must be scaled
% descaledoperator.b_EV  = descaledoperator.b_EV  *gscale;
% descaledoperator.s_IC  = descaledoperator.s_IC  *gscale;
% descaledoperator.s_IV  = descaledoperator.s_IV  *gscale; 
% normA_EC     = norm(descaledoperator.A_EC  ,'fro');
% normA_EV     = norm(descaledoperator.A_EV  ,'fro');
% normA_IC     = norm(descaledoperator.A_IC  ,'fro');
% normA_IV     = norm(descaledoperator.A_IV  ,'fro');
% scaleA_EC    = 1;
% scaleA_EV    = 1;
% scaleA_IC    = 1;
% scaleA_IV    = 1;
% scalec_k     = 1;
% scalecp      = 1;
% scaleca      = 1;
% scalecg      = 1;
% scaleG       = 1;%norm(Gr,'fro');
% maxEscale    = max(normA_EC,normA_EV);
% maxIscale    = max(normA_IC,normA_IV);
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
