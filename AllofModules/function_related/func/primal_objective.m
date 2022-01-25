function [objective] = primal_objective(x_k, Ghat, operators, learningparams, optparams)
    [objective] = 1/2*euclidean_dist_of_x(x_k, Ghat);
end