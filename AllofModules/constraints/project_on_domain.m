function [x_k ] = project_on_domain(x_k, operators)
    x_k.u  = max(x_k.u, operators.domain_of_x_min_x_u);
    x_k.u  = min(x_k.u, operators.domain_of_x_max_x_u);
end