function [x_primalwmst, x_dualwmst, alpha_primalwmst] = getWARMSTARTGlobal(y_EC, y_EV, y_IC, y_IV, x_st_IC, x_st_IV)
global cnstData
global WMST_variables
   pre_nSDP          = cnstData.pre_nSDP;
   nSDP              = cnstData.nSDP;
   n_S               = cnstData.n_S;
   x_primalwmst      =  x_wmst_from_previous(WMST_variables.x_primal,x_st_IC, x_st_IV) ;
   x_dualwmst        = WMST_variables.x_dualwmst    ;
   x_dualwmst.y_EC   = y_EC;
   x_dualwmst.y_EV   = y_EV;
   x_dualwmst.y_IC   = y_IC;
   x_dualwmst.y_IV   = y_IV;
   % map S and Z using extndind map
   [eind_new, eind_pre] = get_map_extndind(cnstData.initL);
   [x_dualwmst.S]    = u_wmst_from_previous(x_dualwmst.S, pre_nSDP, nSDP, n_S, eind_new, eind_pre);
   [x_dualwmst.Z]    = u_wmst_from_previous(x_dualwmst.Z, pre_nSDP, nSDP, n_S, eind_new, eind_pre);
   % delete unwanted indices from alpha
   ind               = [(1:n_S)';eind_new];
   si                = numel(ind);
   alpha_primalwmst  = zeros(si,1);
   alpha_primalwmst(ind) = WMST_variables.alpha_primalwmst([(1:n_S)';eind_pre]) ;
end