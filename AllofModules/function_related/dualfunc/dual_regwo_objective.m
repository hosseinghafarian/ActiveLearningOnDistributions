function [f_dual] = dual_regwo_objective(dualvars_k, x_0, operators, learningparams, optparams)
global cnstData 

    [f_dual] = ...
        dual_regwo_objective_split(dualvars_k.y_EC, dualvars_k.y_EV, dualvars_k.y_IC, dualvars_k.y_IV,...
                                   dualvars_k.S, dualvars_k.Z , dualvars_k.v,...
                                   x_0, operators, learningparams, optparams);
end