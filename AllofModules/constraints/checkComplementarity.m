function [compyAEC, compyAEV,compyAIC,compyAIV,...
          compSDP , compV   ,...
          feasAEC,feasAIC,feasAEV,feasAIV] = checkComplementarity(operators, x_k, dualvars_k)
   n_IC      = operators.n_AIC;
   s_I       = [operators.s_IC;operators.s_IV];
   compSDP   = x_k.u'*dualvars_k.S;
   fvpAEC    = operators.b_EC-operators.A_EC*x_k.u;
   compyAEC  = fvpAEC'*dualvars_k.y_EC;
   fvpAEV    = operators.b_EV-operators.A_EV*x_k.u-operators.B_EV*x_k.w_obeta;
   compyAEV  = fvpAEV'*dualvars_k.y_EV;  
   fvpAIC    = x_k.st(1:n_IC)- operators.A_IC*x_k.u;
   compyAIC  = fvpAIC'*dualvars_k.y_IC;
   fvpAIV    = x_k.st(n_IC+1:end)-operators.A_IV*x_k.u-operators.B_IV*x_k.w_obeta;
   compyAIV  = fvpAIV'*dualvars_k.y_IV;
   vdup      = x_k.st-s_I;
   compV     = vdup'*dualvars_k.v;
   
   feasAEC   = norm(compyAEC);
   feasAIV   = norm(compyAIV);
   feasAEV   = norm(compyAEV);
   feasAIC   = norm(compyAIC);
end
