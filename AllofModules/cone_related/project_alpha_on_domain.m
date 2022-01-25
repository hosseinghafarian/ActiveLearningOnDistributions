function [alpha_k ] = project_alpha_on_domain(alpha_k)
global cnstData
    alpha_k = max(alpha_k, cnstData.lo);
    alpha_k = min(alpha_k, cnstData.up);
end