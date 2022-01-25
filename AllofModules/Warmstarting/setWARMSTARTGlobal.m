function setWARMSTARTGlobal(x_primalwmst, x_dualwmst, alpha_primalwmst, operators)
   global WMST_variables
   WMST_variables.x_primal      = x_primalwmst;
   WMST_variables.n_stIC        = operators.n_AIC;
   WMST_variables.n_stIV        = operators.n_AIV;
   WMST_variables.x_dualwmst    = x_dualwmst;
   WMST_variables.alpha_primalwmst = alpha_primalwmst;
end