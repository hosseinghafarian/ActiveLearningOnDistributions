function [ squares ] = euclidean_dist_of_duals(dualvars_k, dualvars_0)
    squares = norm(dualvars_k.y_EC-dualvars_0.y_EC)^2 + ...
              norm(dualvars_k.y_IC-dualvars_0.y_IC)^2 + ...
              norm(dualvars_k.y_EV-dualvars_0.y_EV)^2 + ...
              norm(dualvars_k.y_IV-dualvars_0.y_IV)^2 + ...
              norm(dualvars_k.S   -dualvars_0.S   )^2 + ...
              norm(dualvars_k.Z   -dualvars_0.Z   )^2 + ...
              norm(dualvars_k.v   -dualvars_0.v   )^2 ;      
end