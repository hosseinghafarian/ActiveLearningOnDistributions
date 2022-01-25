function [x_k, c_k] = project_cone_on_domain(x_k, operators)

    [M, idx] = max(abs(x_k.u));
    if M > abs(operators.domain_of_x_max_x_u(idx))
        c_k = M/abs(operators.domain_of_x_min_x_u(idx));
        x_k.u  = x_k.u/c_k;
    else
        c_k = 1;
    end

    x_k.u = max(x_k.u, operators.domain_of_x_min_x_u);
end