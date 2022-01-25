function [dualvars] = dualvar_conv(y_EC, y_EV, y_IC, y_IV, S, Z ,  v )

    dualvars.y_EC  = y_EC;
    dualvars.y_IC  = y_IC;
    dualvars.y_EV  = y_EV;
    dualvars.y_IV  = y_IV;
    dualvars.S     = S;
    dualvars.Z     = Z;
    dualvars.v     = v;
end